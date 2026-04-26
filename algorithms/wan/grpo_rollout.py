from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RolloutTrace:
    cond: dict[str, Any]
    timesteps: list[torch.Tensor]
    latent_path: list[torch.Tensor]
    old_flow_preds: list[torch.Tensor]
    hist_noise_path: list[torch.Tensor | None]
    decoded_video: torch.Tensor
    reward: torch.Tensor | None = None
    logp_old: torch.Tensor | None = None


@dataclass
class RolloutGroup:
    condition_index: int
    traces: list[RolloutTrace]


def clone_condition_batch(batch: dict[str, Any]) -> dict[str, Any]:
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


@torch.no_grad()
def sample_trace(
    algo,
    batch: dict[str, Any],
    *,
    hist_len: int = 1,
    lang_guidance: float | None = None,
    hist_guidance: float | None = None,
    flow_sampling_noise_std: float = 0.0,
    flow_logprob_sigma: float | None = None,
) -> RolloutTrace:
    """
    Trace-aware variant of Wan sampling.

    This mirrors `WanTextToVideo.sample_seq`, but additionally records:
    - scheduler timesteps
    - latent path
    - old policy flow predictions

    The implementation intentionally stays lightweight and delegates all model-specific
    encoding / scheduler setup to the existing Wan algorithm object.
    """
    if (hist_len - 1) % algo.vae_stride[0] != 0:
        raise ValueError(
            "hist_len - 1 must be a multiple of vae_stride[0] due to temporal vae. "
            f"Got {hist_len} and vae stride {algo.vae_stride[0]}"
        )
    hist_len_lat = (hist_len - 1) // algo.vae_stride[0] + 1

    algo.inference_scheduler, algo.inference_timesteps = algo.build_scheduler(False)
    batch = algo.prepare_embeds(clone_condition_batch(batch))

    clip_embeds = batch["clip_embeds"]
    image_embeds = batch["image_embeds"]
    prompt_embeds = batch["prompt_embeds"]
    video_lat = batch["video_lat"]

    target_dtype = algo.model.patch_embedding.weight.dtype
    if video_lat.dtype != target_dtype:
        video_lat = video_lat.to(target_dtype)
    if clip_embeds is not None and clip_embeds.dtype != target_dtype:
        clip_embeds = clip_embeds.to(target_dtype)
    if image_embeds is not None and image_embeds.dtype != target_dtype:
        image_embeds = image_embeds.to(target_dtype)
    if prompt_embeds is not None:
        prompt_embeds = [u.to(target_dtype) if u.dtype != target_dtype else u for u in prompt_embeds]

    batch_size = video_lat.shape[0]
    if batch_size != 1:
        raise ValueError(f"sample_trace currently expects batch_size=1, got {batch_size}.")

    current_lang_guidance = algo.lang_guidance if lang_guidance is None else float(lang_guidance)
    current_hist_guidance = algo.hist_guidance if hist_guidance is None else float(hist_guidance)
    video_pred_lat = torch.randn_like(video_lat)
    logprob_sigma = float(flow_logprob_sigma) if flow_logprob_sigma is not None else float(flow_sampling_noise_std)

    neg_prompt_embeds = None
    if current_lang_guidance:
        neg_prompt_embeds = algo.encode_text([algo.neg_prompt] * len(batch["prompts"]))
        neg_prompt_embeds = [u.to(target_dtype) if u.dtype != target_dtype else u for u in neg_prompt_embeds]

    latent_path = [video_pred_lat.detach().clone()]
    old_flow_preds: list[torch.Tensor] = []
    hist_noise_path: list[torch.Tensor | None] = []
    timesteps: list[torch.Tensor] = []
    rollout_logp_old = torch.zeros(1, device=video_lat.device, dtype=torch.float32)

    for t in algo.inference_timesteps:
        if algo.diffusion_forcing.enabled:
            video_pred_lat[:, :, :hist_len_lat] = video_lat[:, :, :hist_len_lat]
            t_expanded = torch.full((batch_size, algo.lat_t), t, device=algo.device, dtype=video_pred_lat.dtype)
            t_expanded[:, :hist_len_lat] = algo.inference_timesteps[-1]
        else:
            t_expanded = torch.full((batch_size,), t, device=algo.device, dtype=video_pred_lat.dtype)

        flow_pred = algo.model(
            video_pred_lat,
            t=t_expanded,
            context=prompt_embeds,
            seq_len=algo.max_tokens,
            clip_fea=clip_embeds,
            y=image_embeds,
        )

        if current_lang_guidance:
            no_lang_flow_pred = algo.model(
                video_pred_lat,
                t=t_expanded,
                context=neg_prompt_embeds,
                seq_len=algo.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
            )
        else:
            no_lang_flow_pred = torch.zeros_like(flow_pred)

        if current_hist_guidance and algo.diffusion_forcing.enabled:
            no_hist_video_pred_lat = video_pred_lat.clone()
            hist_noise = torch.randn_like(no_hist_video_pred_lat[:, :, :hist_len_lat])
            no_hist_video_pred_lat[:, :, :hist_len_lat] = hist_noise
            t_hist = t_expanded.clone()
            t_hist[:, :hist_len_lat] = algo.inference_timesteps[0]
            no_hist_flow_pred = algo.model(
                no_hist_video_pred_lat,
                t=t_hist,
                context=prompt_embeds,
                seq_len=algo.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
            )
        else:
            hist_noise = None
            no_hist_flow_pred = torch.zeros_like(flow_pred)

        guided_flow_pred = flow_pred * (1 + current_lang_guidance + current_hist_guidance)
        guided_flow_pred = (
            guided_flow_pred
            - current_lang_guidance * no_lang_flow_pred
            - current_hist_guidance * no_hist_flow_pred
        )

        if flow_sampling_noise_std > 0:
            flow_noise = torch.randn_like(guided_flow_pred) * float(flow_sampling_noise_std)
            sampled_flow_pred = guided_flow_pred + flow_noise
            noise_sq = flow_noise.float().reshape(flow_noise.shape[0], -1).pow(2).mean(dim=1)
            step_logp_old = -0.5 * noise_sq / max(logprob_sigma * logprob_sigma, 1e-8)
            rollout_logp_old = rollout_logp_old + step_logp_old.detach().to(rollout_logp_old.dtype)
        else:
            sampled_flow_pred = guided_flow_pred

        old_flow_preds.append(sampled_flow_pred.detach().clone())
        hist_noise_path.append(hist_noise.detach().clone() if hist_noise is not None else None)
        timesteps.append(t_expanded.detach().clone())
        video_pred_lat = algo.remove_noise(sampled_flow_pred, t, video_pred_lat)
        latent_path.append(video_pred_lat.detach().clone())

    video_pred_lat[:, :, :hist_len_lat] = video_lat[:, :, :hist_len_lat]
    vae_dtype = next(algo.vae.parameters()).dtype
    if video_pred_lat.dtype != vae_dtype:
        video_pred_lat = video_pred_lat.to(vae_dtype)
    decoded_video = algo.decode_video(video_pred_lat)
    decoded_video = decoded_video.permute(0, 2, 1, 3, 4).detach()

    cond = {
        "prompts": list(batch["prompts"]),
        "hist_len": hist_len,
    }
    if "video_path" in batch:
        cond["video_path"] = batch["video_path"]

    return RolloutTrace(
        cond=cond,
        timesteps=timesteps,
        latent_path=latent_path,
        old_flow_preds=old_flow_preds,
        hist_noise_path=hist_noise_path,
        decoded_video=decoded_video,
        logp_old=rollout_logp_old.to(device=decoded_video.device, dtype=decoded_video.dtype),
    )


@torch.no_grad()
def rollout_group(
    algo,
    cond_batch: dict[str, Any],
    *,
    group_size: int,
    hist_len: int = 1,
    flow_sampling_noise_std: float = 0.0,
    flow_logprob_sigma: float | None = None,
) -> list[RolloutGroup]:
    """
    Roll out `group_size` traces per condition.

    Current implementation assumes `cond_batch` has batch size 1 and is called repeatedly by
    an outer trainer. This keeps the first integration simple and explicit.
    """
    groups = []
    for cond_idx in range(group_size):
        trace = sample_trace(
            algo,
            cond_batch,
            hist_len=hist_len,
            flow_sampling_noise_std=flow_sampling_noise_std,
            flow_logprob_sigma=flow_logprob_sigma,
        )
        groups.append(trace)
    return [RolloutGroup(condition_index=0, traces=groups)]
