from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _as_batch(x: torch.Tensor | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if x is None:
        return None
    y = x.to(device=device, dtype=dtype)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    return y


def _match_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    if x.shape[0] == batch_size:
        return x
    if x.shape[0] == 1:
        return x.repeat(batch_size, *([1] * (x.ndim - 1)))
    raise ValueError(f"Batch mismatch: expected {batch_size}, got {x.shape[0]}")


def _weighted_l1(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    if weights is None:
        return x.abs().mean(dim=-1)
    w = weights.to(device=x.device, dtype=x.dtype).flatten().view(1, -1)
    if w.shape[-1] != x.shape[-1]:
        raise ValueError(f"Weight dim mismatch: expected {x.shape[-1]}, got {w.shape[-1]}")
    return (x.abs() * w).sum(dim=-1) / w.sum().clamp_min(1e-6)


@dataclass
class RewardBreakdown:
    reward: torch.Tensor
    valid_mask: torch.Tensor
    feasibility_gate: torch.Tensor
    action_recovery_reward: torch.Tensor
    idm_stability_reward: torch.Tensor
    transition_stability_reward: torch.Tensor
    hard_invalid: torch.Tensor


StepRewardBreakdown = RewardBreakdown


class StepExecutabilityReward:
    """
    Minimal reward for world-model GRPO:
    - feasibility gate
    - action recovery reward
    - IDM stability/confidence reward
    - GT alignment mask on the recovery term
    """

    def __init__(
        self,
        *,
        action_dim: int,
        dof_weights: torch.Tensor | None = None,
        hard_veto_penalty: float = 50.0,
        max_control_delta: float | None = 0.25,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
        feasibility_weight: float = 1.0,
        action_recovery_weight: float = 1.0,
        idm_stability_weight: float = 0.25,
        transition_stability_weight: float = 0.25,
    ):
        self.action_dim = int(action_dim)
        self.dof_weights = dof_weights
        self.hard_veto_penalty = float(hard_veto_penalty)
        self.max_control_delta = max_control_delta
        self.action_low = action_low
        self.action_high = action_high
        self.feasibility_weight = float(feasibility_weight)
        self.action_recovery_weight = float(action_recovery_weight)
        self.idm_stability_weight = float(idm_stability_weight)
        self.transition_stability_weight = float(transition_stability_weight)
        self.last_breakdown: RewardBreakdown | None = None

    def _bounds_violation(self, action_t: torch.Tensor) -> torch.Tensor:
        if self.action_low is None or self.action_high is None:
            return torch.zeros(action_t.shape[0], device=action_t.device, dtype=torch.bool)
        low = self.action_low.to(device=action_t.device, dtype=action_t.dtype).flatten()
        high = self.action_high.to(device=action_t.device, dtype=action_t.dtype).flatten()
        if low.numel() != action_t.shape[-1] or high.numel() != action_t.shape[-1]:
            raise ValueError("Action bounds dimension mismatch")
        below = action_t < low.view(1, -1)
        above = action_t > high.view(1, -1)
        return (below | above).any(dim=1)

    def _delta_violation(self, action_t: torch.Tensor, control_state_t: torch.Tensor | None) -> torch.Tensor:
        if control_state_t is None or self.max_control_delta is None:
            return torch.zeros(action_t.shape[0], device=action_t.device, dtype=torch.bool)
        control = _match_batch(_as_batch(control_state_t, action_t.device, action_t.dtype), action_t.shape[0])
        delta = action_t - control[:, : action_t.shape[-1]]
        return delta.abs().amax(dim=1) > float(self.max_control_delta)

    def score_step(
        self,
        action_t: torch.Tensor,
        *,
        control_state_t: torch.Tensor | None = None,
        prev_actions: torch.Tensor | None = None,
        action_gt: torch.Tensor | None = None,
        action_gt_mask: torch.Tensor | None = None,
        action_samples: torch.Tensor | None = None,
        cond_batch: dict[str, Any] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if action_t.ndim != 2:
            raise ValueError(f"Expected action_t [B, D], got {tuple(action_t.shape)}")
        if action_t.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action_t.shape[-1]}")

        if cond_batch is not None:
            if action_gt is None:
                action_gt = cond_batch.get("action_gt")
            if action_gt_mask is None:
                action_gt_mask = cond_batch.get("action_gt_mask")
            if action_samples is None:
                action_samples = cond_batch.get("idm_action_samples")

        weights = None
        if self.dof_weights is not None:
            weights = self.dof_weights.to(device=action_t.device, dtype=action_t.dtype).flatten()
            if weights.numel() != action_t.shape[-1]:
                raise ValueError(f"dof_weights dim mismatch: expected {action_t.shape[-1]}, got {weights.numel()}")

        non_finite = ~torch.isfinite(action_t).all(dim=1)
        bound_invalid = self._bounds_violation(action_t)
        delta_invalid = self._delta_violation(action_t, control_state_t)
        hard_invalid = non_finite | bound_invalid | delta_invalid
        feasibility_gate = -hard_invalid.float() * self.hard_veto_penalty

        # DEBUG: print veto breakdown every 50 calls to verify normalization alignment
        if not hasattr(self, "_debug_call_count"):
            self._debug_call_count = 0
        self._debug_call_count += 1
        if self._debug_call_count % 10 == 1:
            with torch.no_grad():
                act_min = action_t.min().item()
                act_max = action_t.max().item()
                if control_state_t is not None:
                    ctrl = _as_batch(control_state_t, action_t.device, action_t.dtype)
                    ctrl = _match_batch(ctrl, action_t.shape[0])
                    delta = (action_t - ctrl[:, : action_t.shape[-1]]).abs()
                    delta_max = delta.amax(dim=1)
                    # Show first sample's full 16-dim vector for inspection
                    act_vec = action_t[0].detach().cpu().tolist()
                    act_vec_str = "[" + ", ".join([f"{x:.3f}" for x in act_vec]) + "]"
                    print(
                        f"[REWARD DEBUG call={self._debug_call_count}] "
                        f"shape={list(action_t.shape)} | "
                        f"action_min/max=[{act_min:.3f}, {act_max:.3f}] | "
                        f"action_vec={act_vec_str} | "
                        f"delta_max per sample: {delta_max.tolist()} | "
                        f"hard_invalid={hard_invalid.tolist()}"
                    )
                else:
                    print(
                        f"[REWARD DEBUG call={self._debug_call_count}] "
                        f"action_t [{act_min:.3f}, {act_max:.3f}] | "
                        f"control_state_t=None (delta check skipped) | "
                        f"non_finite={non_finite.tolist()} | "
                        f"hard_invalid={hard_invalid.tolist()}"
                    )

        if action_gt is None:
            action_recovery_reward = torch.zeros(action_t.shape[0], device=action_t.device, dtype=torch.float32)
        else:
            target = _match_batch(_as_batch(action_gt, action_t.device, action_t.dtype), action_t.shape[0])
            recovery = -_weighted_l1(action_t - target[:, : action_t.shape[-1]], weights)
            if action_gt_mask is not None:
                mask = action_gt_mask.to(device=action_t.device, dtype=recovery.dtype).view(-1)
                if mask.numel() == 1 and recovery.shape[0] > 1:
                    mask = mask.repeat(recovery.shape[0])
                action_recovery_reward = recovery * mask
            else:
                action_recovery_reward = recovery

        if action_samples is None:
            idm_stability_reward = torch.zeros(action_t.shape[0], device=action_t.device, dtype=torch.float32)
        else:
            samples = action_samples.to(device=action_t.device, dtype=action_t.dtype)
            if samples.ndim != 3 or samples.shape[-1] != action_t.shape[-1]:
                raise ValueError(f"Expected action_samples [K, B, D], got {tuple(samples.shape)}")
            sample_mean = samples.mean(dim=0, keepdim=True)
            sample_dev = samples - sample_mean
            sample_var = sample_dev.pow(2).mean(dim=0)
            idm_stability_reward = -_weighted_l1(sample_var, weights)

        if prev_actions is None or prev_actions.numel() == 0:
            transition_stability_reward = torch.zeros(action_t.shape[0], device=action_t.device, dtype=torch.float32)
        else:
            prev_actions = prev_actions.to(device=action_t.device, dtype=action_t.dtype)
            prev_last = prev_actions[:, -1]
            transition_stability_reward = -_weighted_l1(action_t - prev_last, weights)

        reward = (
            self.feasibility_weight * feasibility_gate
            + self.action_recovery_weight * action_recovery_reward
            + self.idm_stability_weight * idm_stability_reward
            + self.transition_stability_weight * transition_stability_reward
        )

        valid_mask = (
            action_gt_mask.to(device=action_t.device, dtype=action_t.dtype).view(-1)
            if action_gt_mask is not None
            else torch.ones(action_t.shape[0], device=action_t.device, dtype=action_t.dtype)
        )
        self.last_breakdown = RewardBreakdown(
            reward=reward.detach(),
            valid_mask=valid_mask.detach(),
            feasibility_gate=feasibility_gate.detach(),
            action_recovery_reward=action_recovery_reward.detach(),
            idm_stability_reward=idm_stability_reward.detach(),
            transition_stability_reward=transition_stability_reward.detach(),
            hard_invalid=hard_invalid.detach().float(),
        )
        return reward

    def score_next_action(
        self,
        action_t: torch.Tensor,
        *,
        control_state_t: torch.Tensor | None = None,
        cond_batch: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        cond_batch = cond_batch or {}
        return self.score_step(
            action_t,
            control_state_t=control_state_t,
            action_gt=cond_batch.get("action_gt"),
            action_gt_mask=cond_batch.get("action_gt_mask"),
            action_samples=cond_batch.get("idm_action_samples"),
        )
