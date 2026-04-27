from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .depth_backends import DepthBackend
from .idm_backends import IDMBackend
from .state_action_reward import StepExecutabilityReward
from .state_unrolled_grpo import rollout_state_unrolled_group
from .state_unrolled_grpo_objective import (
    StateUnrolledGRPOConfig,
    StateUnrolledGRPOLoss,
    compute_single_trace_grpo_loss,
    group_normalize_rewards,
)


@dataclass
class StateUnrolledGRPOTrainerConfig:
    group_size: int = 4
    horizon_steps: int = 4
    hist_len: int = 1
    flow_sampling_noise_std: float = 0.05
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    surrogate_sigma: float = 1.0
    log_ratio_clip: float = 5.0
    grad_clip_norm: float | None = 1.0
    ref_update_interval: int = 0
    discount_gamma: float = 1.0
    num_inner_epochs: int = 1


class StateUnrolledGRPOTrainer:
    def __init__(
        self,
        *,
        policy_algo,
        ref_algo,
        optimizer: torch.optim.Optimizer,
        idm_backend: IDMBackend,
        depth_backend: DepthBackend,
        reward_model: StepExecutabilityReward,
        cfg: StateUnrolledGRPOTrainerConfig | None = None,
    ):
        self.policy_algo = policy_algo
        self.ref_algo = ref_algo
        self.optimizer = optimizer
        self.idm_backend = idm_backend
        self.depth_backend = depth_backend
        self.reward_model = reward_model
        self.cfg = cfg or StateUnrolledGRPOTrainerConfig()
        self.step = 0

    def sync_reference_policy(self) -> None:
        if self.ref_algo is None:
            return
        self.ref_algo.load_state_dict(self.policy_algo.state_dict(), strict=False)
        self.ref_algo.eval()
        for param in self.ref_algo.parameters():
            param.requires_grad_(False)

    def _compute_grad_norm(self, device, dtype) -> torch.Tensor:
        grad_norm = torch.zeros(1, device=device, dtype=dtype)
        if self.cfg.grad_clip_norm is not None:
            val = torch.nn.utils.clip_grad_norm_(
                self.policy_algo.parameters(), self.cfg.grad_clip_norm
            )
            grad_norm = torch.as_tensor(val, device=device, dtype=dtype)
        else:
            total_sq = None
            for param in self.policy_algo.parameters():
                if param.grad is None:
                    continue
                g = param.grad.detach().float().pow(2).sum()
                total_sq = g if total_sq is None else total_sq + g
            if total_sq is not None:
                grad_norm = total_sq.sqrt().to(device=device, dtype=dtype)
        return grad_norm

    def train_step(self, cond_batch: dict[str, Any]) -> StateUnrolledGRPOLoss:
        # ── 1. Rollout (no-grad, sequential, OK on memory) ───────────────────
        rollout_groups = rollout_state_unrolled_group(
            self.policy_algo,
            cond_batch,
            idm_backend=self.idm_backend,
            depth_backend=self.depth_backend,
            reward_model=self.reward_model,
            group_size=self.cfg.group_size,
            horizon_steps=self.cfg.horizon_steps,
            hist_len=self.cfg.hist_len,
            flow_sampling_noise_std=self.cfg.flow_sampling_noise_std,
            flow_logprob_sigma=self.cfg.surrogate_sigma,
            discount_gamma=self.cfg.discount_gamma,
        )

        grpo_cfg = StateUnrolledGRPOConfig(
            clip_eps=self.cfg.clip_eps,
            beta_kl=self.cfg.beta_kl,
            surrogate_sigma=self.cfg.surrogate_sigma,
            log_ratio_clip=self.cfg.log_ratio_clip,
        )

        # ── 2. Pre-compute advantages (no-grad) ──────────────────────────────
        # Done ONCE outside the inner loop so advantages are stable across epochs.
        algo_device = next(self.policy_algo.parameters()).device
        group_advantages: list[torch.Tensor] = []
        group_rewards_mean: list[torch.Tensor] = []
        group_rewards_std: list[torch.Tensor] = []
        for group in rollout_groups:
            rewards = (
                torch.stack([t.cumulative_reward for t in group.traces], dim=0)
                .to(algo_device)
                .view(-1)
            )
            group_rewards_mean.append(rewards.mean().detach())
            group_rewards_std.append(rewards.std(unbiased=False).detach())
            group_advantages.append(group_normalize_rewards(rewards).detach())

        total_traces = sum(len(g.traces) for g in rollout_groups)
        last_loss_dict: StateUnrolledGRPOLoss | None = None

        # ── 3. Inner epochs ──────────────────────────────────────────────────
        for _inner_epoch in range(self.cfg.num_inner_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            agg_loss = torch.zeros([], device=algo_device)
            agg_policy = torch.zeros([], device=algo_device)
            agg_kl = torch.zeros([], device=algo_device)
            agg_ratio = torch.zeros([], device=algo_device)
            agg_advantage = torch.zeros([], device=algo_device)
            n_valid = 0

            for group_i, group in enumerate(rollout_groups):
                advantages = group_advantages[group_i]
                for trace, adv in zip(group.traces, advantages):
                    total_loss, policy_loss, kl_loss, ratio = compute_single_trace_grpo_loss(
                        algo=self.policy_algo,
                        ref_algo=self.ref_algo,
                        trace=trace,
                        advantage=adv,
                        cfg=grpo_cfg,
                    )
                    # Scale by 1/total so accumulated grad ≈ batch mean
                    (total_loss / total_traces).backward()

                    agg_loss = agg_loss + total_loss.detach()
                    agg_policy = agg_policy + policy_loss.detach()
                    agg_kl = agg_kl + kl_loss.detach()
                    agg_ratio = agg_ratio + ratio.detach()
                    agg_advantage = agg_advantage + adv.detach()
                    n_valid += 1

            if n_valid == 0:
                continue

            grad_norm = self._compute_grad_norm(algo_device, agg_loss.dtype)
            self.optimizer.step()

            last_loss_dict = StateUnrolledGRPOLoss(
                loss=agg_loss / n_valid,
                policy_loss=agg_policy / n_valid,
                kl_loss=agg_kl / n_valid,
                mean_ratio=agg_ratio / n_valid,
                mean_advantage=agg_advantage / n_valid,
                group_reward_mean=torch.stack(group_rewards_mean).mean(),
                group_reward_std=torch.stack(group_rewards_std).mean(),
                grad_norm=grad_norm,
            )

        self.step += 1
        if self.cfg.ref_update_interval and self.step % self.cfg.ref_update_interval == 0:
            self.sync_reference_policy()

        return last_loss_dict

