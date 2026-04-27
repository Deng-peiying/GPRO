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
    compute_state_unrolled_grpo_loss,
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

    def train_step(self, cond_batch: dict[str, Any]) -> StateUnrolledGRPOLoss:
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
        last_loss_dict = None
        for inner_epoch in range(self.cfg.num_inner_epochs):
            loss_dict = compute_state_unrolled_grpo_loss(
                algo=self.policy_algo,
                ref_algo=self.ref_algo,
                rollout_groups=rollout_groups,
                cfg=StateUnrolledGRPOConfig(
                    clip_eps=self.cfg.clip_eps,
                    beta_kl=self.cfg.beta_kl,
                    surrogate_sigma=self.cfg.surrogate_sigma,
                    log_ratio_clip=self.cfg.log_ratio_clip,
                ),
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss_dict.loss.backward()
            grad_norm = torch.zeros(1, device=loss_dict.loss.device, dtype=loss_dict.loss.dtype)
            if self.cfg.grad_clip_norm is not None:
                grad_norm_value = torch.nn.utils.clip_grad_norm_(self.policy_algo.parameters(), self.cfg.grad_clip_norm)
                grad_norm = torch.as_tensor(grad_norm_value, device=loss_dict.loss.device, dtype=loss_dict.loss.dtype)
            else:
                total_sq = None
                for param in self.policy_algo.parameters():
                    if param.grad is None:
                        continue
                    grad_sq = param.grad.detach().float().pow(2).sum()
                    total_sq = grad_sq if total_sq is None else total_sq + grad_sq
                if total_sq is not None:
                    grad_norm = total_sq.sqrt().to(device=loss_dict.loss.device, dtype=loss_dict.loss.dtype)
            loss_dict.grad_norm = grad_norm
            self.optimizer.step()
            last_loss_dict = loss_dict

        self.step += 1
        if self.cfg.ref_update_interval and self.step % self.cfg.ref_update_interval == 0:
            self.sync_reference_policy()
        return last_loss_dict
