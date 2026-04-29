from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .depth_backends import DepthBackend
from .idm_backends import IDMBackend, decode_next_action
from .grpo_rollout import RolloutTrace, sample_trace
from .state_action_reward import StepExecutabilityReward


@dataclass
class StepTransition:
    rollout_trace: RolloutTrace
    step_cond: dict[str, Any]
    current_rgb: torch.Tensor
    current_depth: torch.Tensor
    next_rgb: torch.Tensor
    next_depth: torch.Tensor
    action_t: torch.Tensor
    reward_t: torch.Tensor
    training_mask_t: torch.Tensor
    control_state_t: torch.Tensor
    control_state_t1: torch.Tensor


@dataclass
class StateUnrolledTrace:
    transitions: list[StepTransition]
    cumulative_reward: torch.Tensor
    effective_horizon: torch.Tensor
    final_control_state: torch.Tensor


@dataclass
class StateUnrolledGroup:
    condition_index: int
    traces: list[StateUnrolledTrace]


def _clone_cond_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        else:
            cloned[key] = value
    return cloned


def _ensure_batch_rgb(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim == 4:
        return rgb
    if rgb.ndim == 3:
        return rgb.unsqueeze(0)
    raise ValueError(f"Expected RGB [B, C, H, W] or [C, H, W], got {tuple(rgb.shape)}")


def _ensure_batch_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3:
        return depth
    if depth.ndim == 2:
        return depth.unsqueeze(0)
    raise ValueError(f"Expected depth [B, H, W] or [H, W], got {tuple(depth.shape)}")


def _repeat_first_frame(frame: torch.Tensor, n_frames: int) -> torch.Tensor:
    return frame.unsqueeze(1).repeat(1, n_frames, 1, 1, 1)


def _build_history_video(history_rgb: list[torch.Tensor], n_frames: int) -> torch.Tensor:
    if not history_rgb:
        raise ValueError("history_rgb must contain at least one frame")
    if len(history_rgb) >= n_frames:
        return torch.stack(history_rgb[-n_frames:], dim=1)
    video = torch.stack(history_rgb, dim=1)
    pad = history_rgb[-1].unsqueeze(1).repeat(1, n_frames - len(history_rgb), 1, 1, 1)
    return torch.cat([video, pad], dim=1)


def _prepare_step_condition(
    cond_batch: dict[str, Any],
    *,
    history_rgb: list[torch.Tensor],
    prompt_list: list[str],
    n_frames: int,
    step_index: int,
) -> dict[str, Any]:
    step_batch = _clone_cond_batch(cond_batch)
    current_rgb = history_rgb[-1]
    step_batch["videos"] = _build_history_video(history_rgb, n_frames)
    step_batch["prompts"] = prompt_list
    target_videos = cond_batch.get("target_videos")
    actual_frames = cond_batch.get("actual_frames")
    if isinstance(actual_frames, torch.Tensor):
        actual_frames_value = int(actual_frames.view(-1)[0].item())
    else:
        actual_frames_value = None
    if isinstance(target_videos, torch.Tensor) and target_videos.ndim == 5:
        hist_frames = min(len(history_rgb), n_frames)
        target_idx = min(hist_frames, target_videos.shape[1] - 1)
        valid_rgb = target_idx < target_videos.shape[1]
        if actual_frames_value is not None:
            valid_rgb = hist_frames < actual_frames_value
        step_batch["action_gt_mask"] = torch.tensor([1.0 if valid_rgb else 0.0], device=current_rgb.device)
        if valid_rgb:
            step_batch["target_rgb"] = target_videos[:, target_idx]
        else:
            step_batch["target_rgb"] = target_videos[:, -1]
    else:
        step_batch["action_gt_mask"] = torch.tensor([0.0], device=current_rgb.device)

    depth_video = cond_batch.get("depth_video")
    if isinstance(depth_video, torch.Tensor) and depth_video.ndim == 4 and depth_video.shape[1] > 0:
        target_idx = min(min(len(history_rgb), n_frames), depth_video.shape[1] - 1)
        step_batch["target_depth"] = depth_video[:, target_idx]

    action_gt_seq = cond_batch.get("action_seq")
    action_gt_mask_seq = cond_batch.get("action_seq_mask")
    if isinstance(action_gt_seq, torch.Tensor) and action_gt_seq.ndim == 3 and action_gt_seq.shape[1] > step_index:
        step_batch["action_gt"] = action_gt_seq[:, step_index]
        if (
            isinstance(action_gt_mask_seq, torch.Tensor)
            and action_gt_mask_seq.ndim == 2
            and action_gt_mask_seq.shape[1] > step_index
        ):
            step_batch["action_gt_mask"] = action_gt_mask_seq[:, step_index]
    elif "action" in cond_batch:
        step_batch["action_gt"] = cond_batch["action"]
        single_step_mask = 1.0 if step_index == 0 else 0.0
        step_batch["action_gt_mask"] = torch.tensor([single_step_mask], device=current_rgb.device)
    return step_batch


def _detach_condition(batch: dict[str, Any]) -> dict[str, Any]:
    detached = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            detached[key] = value.detach().clone()
        elif isinstance(value, list):
            detached[key] = list(value)
        elif isinstance(value, tuple):
            detached[key] = tuple(value)
        else:
            detached[key] = value
    return detached


def _backend_device(backend: Any, fallback: torch.device) -> torch.device:
    device = getattr(backend, "device", None)
    if device is None:
        return fallback
    return torch.device(device)


def _select_depth_condition(cond_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    selected = {}
    for key in ("pred_depth_video", "depth_video", "current_depth", "depth"):
        if key in cond_batch:
            value = cond_batch[key]
            selected[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return selected


def _select_idm_condition(cond_batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    selected = {}
    for key, value in cond_batch.items():
        if isinstance(value, torch.Tensor):
            if key in {"control_state", "current_depth", "pred_depth_video", "depth_video", "left_endpose", "right_endpose", "ee_xyz"}:
                selected[key] = value.to(device)
        else:
            if key in {"prompts", "task_name", "episode_path", "step"}:
                selected[key] = value
    return selected


def _build_idm_action_samples(
    idm_backend: IDMBackend,
    *,
    current_rgb: torch.Tensor,
    next_rgb: torch.Tensor,
    control_state: torch.Tensor,
    current_depth: torch.Tensor,
    next_depth: torch.Tensor,
    cond_batch: dict[str, Any],
    num_samples: int = 3,
    rgb_noise_std: float = 0.01,
    depth_noise_std: float = 0.005,
) -> torch.Tensor:
    samples = []
    for _ in range(num_samples):
        noisy_next_rgb = (next_rgb + torch.randn_like(next_rgb) * rgb_noise_std).clamp(-1.0, 1.0)
        noisy_next_depth = (next_depth + torch.randn_like(next_depth) * depth_noise_std).clamp(0.0, 1.0)
        action = decode_next_action(
            idm_backend,
            current_rgb=current_rgb,
            next_rgb=noisy_next_rgb,
            control_state=control_state,
            current_depth=current_depth,
            next_depth=noisy_next_depth,
            cond_batch=cond_batch,
        ).float()
        samples.append(action)
    return torch.stack(samples, dim=0)


@torch.no_grad()
def rollout_state_unrolled_trace(
    algo,
    cond_batch: dict[str, Any],
    *,
    idm_backend: IDMBackend,
    depth_backend: DepthBackend,
    reward_model: StepExecutabilityReward,
    horizon_steps: int,
    hist_len: int = 1,
    flow_sampling_noise_std: float = 0.0,
    flow_logprob_sigma: float | None = None,
    discount_gamma: float = 1.0,
) -> StateUnrolledTrace:
    if "current_rgb" not in cond_batch:
        raise KeyError("cond_batch must contain current_rgb for state-unrolled rollout")
    if "current_depth" not in cond_batch:
        raise KeyError("cond_batch must contain current_depth for state-unrolled rollout")
    if "control_state" not in cond_batch:
        raise KeyError("cond_batch must contain control_state for state-unrolled rollout")

    current_rgb = _ensure_batch_rgb(cond_batch["current_rgb"]).to(algo.device)
    current_depth = _ensure_batch_depth(cond_batch["current_depth"]).to(algo.device, dtype=torch.float32)
    control_state = cond_batch["control_state"].to(algo.device, dtype=torch.float32)
    depth_device = _backend_device(depth_backend, algo.device)
    idm_device = _backend_device(idm_backend, algo.device)
    prompt_list = list(cond_batch.get("prompts", [""]))
    if len(prompt_list) != current_rgb.shape[0]:
        if len(prompt_list) == 1:
            prompt_list = prompt_list * current_rgb.shape[0]
        else:
            raise ValueError("prompts batch size mismatch")

    transitions: list[StepTransition] = []
    cumulative_reward = torch.zeros(current_rgb.shape[0], device=algo.device, dtype=torch.float32)
    effective_horizon = torch.zeros(current_rgb.shape[0], device=algo.device, dtype=torch.float32)
    prev_actions = None
    history_rgb = [current_rgb]

    for step_index in range(horizon_steps):
        step_cond = _prepare_step_condition(
            cond_batch,
            history_rgb=history_rgb,
            prompt_list=prompt_list,
            n_frames=algo.n_frames,
            step_index=step_index,
        )
        step_cond["current_rgb"] = current_rgb
        step_cond["current_depth"] = current_depth
        step_cond["control_state"] = control_state

        rollout_trace = sample_trace(
            algo,
            step_cond,
            hist_len=hist_len,
            flow_sampling_noise_std=flow_sampling_noise_std,
            flow_logprob_sigma=flow_logprob_sigma,
        )
        predicted_video = rollout_trace.decoded_video
        next_rgb = predicted_video[:, hist_len: hist_len + 1]
        if next_rgb.shape[1] != 1:
            raise ValueError(f"Expected one next frame, got shape {tuple(next_rgb.shape)}")
        depth_cond = _select_depth_condition(step_cond, depth_device)
        if "current_depth" not in depth_cond:
            depth_cond["current_depth"] = current_depth.to(depth_device)
        next_depth = depth_backend.predict_depth(next_rgb.to(depth_device), depth_cond)
        if next_depth.ndim != 4 or next_depth.shape[1] != 1:
            raise ValueError(f"Expected next depth [B, 1, H, W], got {tuple(next_depth.shape)}")
        next_depth_wan = next_depth.to(algo.device)

        idm_cond = _select_idm_condition(step_cond, idm_device)
        action_t = decode_next_action(
            idm_backend,
            current_rgb=current_rgb.unsqueeze(1).to(idm_device),
            next_rgb=next_rgb.to(idm_device),
            control_state=control_state.to(idm_device),
            current_depth=current_depth.unsqueeze(1).to(idm_device),
            next_depth=next_depth.to(idm_device),
            cond_batch=idm_cond,
        ).float().to(algo.device)

        idm_action_samples = _build_idm_action_samples(
            idm_backend,
            current_rgb=current_rgb.unsqueeze(1).to(idm_device),
            next_rgb=next_rgb.to(idm_device),
            control_state=control_state.to(idm_device),
            current_depth=current_depth.unsqueeze(1).to(idm_device),
            next_depth=next_depth.to(idm_device),
            cond_batch=idm_cond,
        ).to(algo.device)
        step_cond["idm_action_samples"] = idm_action_samples

        reward_t = reward_model.score_step(
            action_t,
            control_state_t=control_state,
            prev_actions=prev_actions,
            cond_batch=step_cond,
        )
        training_mask_t = step_cond.get("action_gt_mask")
        if training_mask_t is None:
            training_mask_t = torch.ones(action_t.shape[0], device=algo.device, dtype=torch.float32)
        else:
            training_mask_t = training_mask_t.to(device=algo.device, dtype=torch.float32).view(-1)
        control_state_t1 = action_t

        transitions.append(
            StepTransition(
                rollout_trace=rollout_trace,
                step_cond=_detach_condition(step_cond),
                current_rgb=current_rgb.detach().clone(),
                current_depth=current_depth.detach().clone(),
                next_rgb=next_rgb[:, 0].detach().clone(),
                next_depth=next_depth_wan[:, 0].detach().clone(),
                action_t=action_t.detach().clone(),
                reward_t=reward_t.detach().clone(),
                training_mask_t=training_mask_t.detach().clone(),
                control_state_t=control_state.detach().clone(),
                control_state_t1=control_state_t1.detach().clone(),
            )
        )
        # Exponential discount: gamma^t * r_t
        discount = discount_gamma ** step_index
        cumulative_reward = cumulative_reward + discount * reward_t * training_mask_t
        effective_horizon = effective_horizon + training_mask_t

        prev_actions = (
            action_t.unsqueeze(1)
            if prev_actions is None
            else torch.cat([prev_actions, action_t.unsqueeze(1)], dim=1)
        )
        current_rgb = next_rgb[:, 0]
        current_depth = next_depth_wan[:, 0]
        control_state = control_state_t1
        history_rgb.append(current_rgb)

    return StateUnrolledTrace(
        transitions=transitions,
        cumulative_reward=cumulative_reward,
        effective_horizon=effective_horizon,
        final_control_state=control_state.detach().clone(),
    )


@torch.no_grad()
def rollout_state_unrolled_group(
    algo,
    cond_batch: dict[str, Any],
    *,
    idm_backend: IDMBackend,
    depth_backend: DepthBackend,
    reward_model: StepExecutabilityReward,
    group_size: int,
    horizon_steps: int,
    hist_len: int = 1,
    flow_sampling_noise_std: float = 0.0,
    flow_logprob_sigma: float | None = None,
    discount_gamma: float = 1.0,
) -> list[StateUnrolledGroup]:
    traces = []
    for _ in range(group_size):
        traces.append(
            rollout_state_unrolled_trace(
                algo,
                cond_batch,
                idm_backend=idm_backend,
                depth_backend=depth_backend,
                reward_model=reward_model,
                horizon_steps=horizon_steps,
                hist_len=hist_len,
                flow_sampling_noise_std=flow_sampling_noise_std,
                flow_logprob_sigma=flow_logprob_sigma,
                discount_gamma=discount_gamma,
            )
        )
    return [StateUnrolledGroup(condition_index=0, traces=traces)]
