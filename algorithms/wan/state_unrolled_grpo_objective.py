from __future__ import annotations

from dataclasses import dataclass

import torch

from .flow_grpo_objective import _compute_trace_logprob_surrogate
from .state_unrolled_grpo import StateUnrolledGroup


@dataclass
class StateUnrolledGRPOConfig:
    clip_eps: float = 0.2
    beta_kl: float = 0.01
    surrogate_sigma: float = 1.0
    eps: float = 1e-6
    log_ratio_clip: float = 5.0


@dataclass
class StateUnrolledGRPOLoss:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    mean_ratio: torch.Tensor
    mean_advantage: torch.Tensor
    group_reward_mean: torch.Tensor
    group_reward_std: torch.Tensor
    grad_norm: torch.Tensor | None = None


def _algo_device(algo) -> torch.device:
    """Return the device of the main DiT model (not auxiliary sub-models like VAE/CLIP)."""
    if hasattr(algo, "model") and hasattr(algo.model, "parameters"):
        return next(algo.model.parameters()).device
    return next(algo.parameters()).device


def _move_cond_batch(cond_batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in cond_batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = list(value)
        elif isinstance(value, tuple):
            moved[key] = tuple(value)
        else:
            moved[key] = value
    return moved


def group_normalize_rewards(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def compute_single_trace_grpo_loss(
    *,
    algo,
    ref_algo,
    trace,
    advantage: torch.Tensor,
    cfg: "StateUnrolledGRPOConfig",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GRPO loss for a single rollout trace.

    Returns (total_loss, policy_loss, kl_loss, ratio) — all scalar tensors.
    Caller is responsible for scaling by 1/N before calling .backward().
    """
    algo_device = _algo_device(algo)
    logp_now_steps: list[torch.Tensor] = []
    logp_old_steps: list[torch.Tensor] = []

    for transition in trace.transitions:
        step_mask = transition.training_mask_t.to(algo_device, dtype=torch.float32).view(-1)
        if step_mask.max().item() <= 0:
            continue
        step_cfg = type("FlowLikeCfg", (), {"surrogate_sigma": cfg.surrogate_sigma})()
        logp_now, _ = _compute_trace_logprob_surrogate(
            algo,
            transition.rollout_trace,
            cond_batch=_move_cond_batch(transition.step_cond, algo_device),
            cfg=step_cfg,
        )
        logp_old = transition.rollout_trace.logp_old.to(device=logp_now.device, dtype=logp_now.dtype)
        logp_now_steps.append(logp_now * step_mask.mean())
        logp_old_steps.append(logp_old.squeeze(0) * step_mask.mean())

    if not logp_now_steps:
        zero = torch.zeros([], device=algo_device)
        return zero, zero, zero, torch.ones([], device=algo_device)

    logp_now_total = torch.stack(logp_now_steps).sum()
    logp_old_total = torch.stack(logp_old_steps).sum()

    logp_ref_total = None
    if ref_algo is not None:
        ref_device = _algo_device(ref_algo)
        logp_ref_steps: list[torch.Tensor] = []
        for transition in trace.transitions:
            step_mask = transition.training_mask_t.to(ref_device, dtype=torch.float32).view(-1)
            if step_mask.max().item() <= 0:
                continue
            step_cfg = type("FlowLikeCfg", (), {"surrogate_sigma": cfg.surrogate_sigma})()
            logp_ref, _ = _compute_trace_logprob_surrogate(
                ref_algo,
                transition.rollout_trace,
                cond_batch=_move_cond_batch(transition.step_cond, ref_device),
                cfg=step_cfg,
            )
            logp_ref_steps.append(
                logp_ref.to(logp_now_total.device) * step_mask.mean().to(logp_now_total.device)
            )
        if logp_ref_steps:
            logp_ref_total = torch.stack(logp_ref_steps).sum()

    log_ratio = (logp_now_total - logp_old_total.detach()).clamp(
        min=-float(cfg.log_ratio_clip), max=float(cfg.log_ratio_clip)
    )
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
    policy_loss = -torch.minimum(unclipped, clipped)

    if logp_ref_total is None:
        kl_loss = torch.zeros_like(policy_loss)
    else:
        kl_delta = logp_now_total - logp_ref_total.detach()
        kl_loss = cfg.beta_kl * kl_delta.pow(2)

    total_loss = policy_loss + kl_loss
    return total_loss, policy_loss, kl_loss, ratio


def compute_state_unrolled_grpo_loss(
    *,
    algo,
    ref_algo,
    rollout_groups: list[StateUnrolledGroup],
    cfg: StateUnrolledGRPOConfig,
) -> StateUnrolledGRPOLoss:
    losses = []
    policy_losses = []
    kl_losses = []
    ratios = []
    mean_advantages = []
    group_reward_means = []
    group_reward_stds = []

    for group in rollout_groups:
        algo_device = _algo_device(algo)
        rewards = torch.stack([trace.cumulative_reward for trace in group.traces], dim=0).to(algo_device).view(-1)
        group_reward_means.append(rewards.mean())
        group_reward_stds.append(rewards.std(unbiased=False))
        advantages = group_normalize_rewards(rewards, eps=cfg.eps)
        mean_advantages.append(advantages.mean())

        for trace, advantage in zip(group.traces, advantages):
            logp_now_steps = []
            logp_old_steps = []

            for transition in trace.transitions:
                step_mask = transition.training_mask_t.to(algo_device, dtype=torch.float32).view(-1)
                if step_mask.max().item() <= 0:
                    continue
                step_cfg = type(
                    "FlowLikeCfg",
                    (),
                    {
                        "surrogate_sigma": cfg.surrogate_sigma,
                    },
                )()
                logp_now, _ = _compute_trace_logprob_surrogate(
                    algo,
                    transition.rollout_trace,
                    cond_batch=_move_cond_batch(transition.step_cond, algo_device),
                    cfg=step_cfg,
                )
                logp_old = transition.rollout_trace.logp_old.to(device=logp_now.device, dtype=logp_now.dtype)
                logp_now_steps.append(logp_now * step_mask.mean())
                logp_old_steps.append(logp_old.squeeze(0) * step_mask.mean())

            if not logp_now_steps:
                continue

            logp_now_total = torch.stack(logp_now_steps).sum()
            logp_old_total = torch.stack(logp_old_steps).sum()
            if ref_algo is not None:
                ref_device = _algo_device(ref_algo)
                logp_ref_steps = []
                for transition in trace.transitions:
                    step_mask = transition.training_mask_t.to(ref_device, dtype=torch.float32).view(-1)
                    if step_mask.max().item() <= 0:
                        continue
                    step_cfg = type(
                        "FlowLikeCfg",
                        (),
                        {
                            "surrogate_sigma": cfg.surrogate_sigma,
                        },
                    )()
                    logp_ref, _ = _compute_trace_logprob_surrogate(
                        ref_algo,
                        transition.rollout_trace,
                        cond_batch=_move_cond_batch(transition.step_cond, ref_device),
                        cfg=step_cfg,
                    )
                    logp_ref_steps.append(logp_ref.to(logp_now_total.device) * step_mask.mean().to(logp_now_total.device))
                if not logp_ref_steps:
                    continue
                logp_ref_total = torch.stack(logp_ref_steps).sum()
            else:
                logp_ref_total = None

            log_ratio = (logp_now_total - logp_old_total.detach()).clamp(
                min=-float(cfg.log_ratio_clip),
                max=float(cfg.log_ratio_clip),
            )
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantage
            policy_loss = -torch.minimum(unclipped, clipped)
            if logp_ref_total is None:
                kl_loss = torch.zeros_like(policy_loss)
            else:
                kl_delta = logp_now_total - logp_ref_total.detach()
                kl_loss = cfg.beta_kl * kl_delta.pow(2)
            total_loss = policy_loss + kl_loss

            losses.append(total_loss)
            policy_losses.append(policy_loss)
            kl_losses.append(kl_loss)
            ratios.append(ratio)

    return StateUnrolledGRPOLoss(
        loss=torch.stack(losses).mean(),
        policy_loss=torch.stack(policy_losses).mean(),
        kl_loss=torch.stack(kl_losses).mean(),
        mean_ratio=torch.stack(ratios).mean(),
        mean_advantage=torch.stack(mean_advantages).mean(),
        group_reward_mean=torch.stack(group_reward_means).mean(),
        group_reward_std=torch.stack(group_reward_stds).mean(),
    )
