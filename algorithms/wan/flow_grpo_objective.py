from __future__ import annotations

from typing import Any

import torch

from .grpo_rollout import RolloutTrace


def _compute_surrogate_step_logprob(
    flow_pred: torch.Tensor,
    target_flow: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    diff = (flow_pred.float() - target_flow.float()).reshape(flow_pred.shape[0], -1)
    # Use per-element mean squared error so the surrogate scale does not explode
    # with latent resolution or frame count.
    return -0.5 * torch.mean(diff * diff, dim=1) / max(sigma * sigma, 1e-8)


def _recompute_flow_predictions(
    algo,
    trace: RolloutTrace,
    *,
    cond_batch: dict[str, Any],
) -> list[torch.Tensor]:
    """
    Recompute flow predictions for the stored latent path under the current model.

    Current implementation uses the recorded latent path and the same condition batch.
    It assumes the trace was generated from a batch size of 1.
    """
    if not hasattr(algo, "inference_timesteps") or algo.inference_timesteps is None:
        algo.inference_scheduler, algo.inference_timesteps = algo.build_scheduler(False)

    batch = algo.prepare_embeds(algo.clone_batch(cond_batch))
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

    hist_len = int(trace.cond.get("hist_len", 1))
    hist_len_lat = (hist_len - 1) // algo.vae_stride[0] + 1
    current_lang_guidance = float(getattr(algo, "lang_guidance", 0.0))
    current_hist_guidance = float(getattr(algo, "hist_guidance", 0.0))

    neg_prompt_embeds = None
    if current_lang_guidance:
        neg_prompt_embeds = algo.encode_text([algo.neg_prompt] * len(batch["prompts"]))
        neg_prompt_embeds = [u.to(target_dtype) if u.dtype != target_dtype else u for u in neg_prompt_embeds]

    current_preds: list[torch.Tensor] = []
    hist_noise_path = getattr(trace, "hist_noise_path", None)

    for step_idx, (x_k, t_k) in enumerate(zip(trace.latent_path[:-1], trace.timesteps)):
        video_pred_lat = x_k.to(dtype=target_dtype, device=algo.device)
        t_expanded = t_k.to(device=algo.device, dtype=video_pred_lat.dtype)

        if getattr(algo, "diffusion_forcing", None) is not None and algo.diffusion_forcing.enabled:
            video_pred_lat = video_pred_lat.clone()
            video_pred_lat[:, :, :hist_len_lat] = video_lat[:, :, :hist_len_lat]

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

        if current_hist_guidance and getattr(algo, "diffusion_forcing", None) is not None and algo.diffusion_forcing.enabled:
            no_hist_video_pred_lat = video_pred_lat.clone()
            hist_noise = None
            if hist_noise_path is not None and step_idx < len(hist_noise_path):
                hist_noise = hist_noise_path[step_idx]
            if hist_noise is None:
                hist_noise = torch.zeros_like(no_hist_video_pred_lat[:, :, :hist_len_lat])
            hist_noise = hist_noise.to(device=algo.device, dtype=video_pred_lat.dtype)
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
            no_hist_flow_pred = torch.zeros_like(flow_pred)

        guided_flow_pred = flow_pred * (1 + current_lang_guidance + current_hist_guidance)
        guided_flow_pred = (
            guided_flow_pred
            - current_lang_guidance * no_lang_flow_pred
            - current_hist_guidance * no_hist_flow_pred
        )
        current_preds.append(guided_flow_pred)
    return current_preds


def _compute_trace_logprob_surrogate(
    algo,
    trace: RolloutTrace,
    *,
    cond_batch: dict[str, Any],
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
    - surrogate logp under the queried policy
    - self-consistency baseline logp

    For the first implementation, the "target flow" is the stored old flow prediction.
    This keeps the objective stable and practical for a first Flow-GRPO iteration.
    """
    current_preds = _recompute_flow_predictions(algo, trace, cond_batch=cond_batch)
    step_logps = []
    ref_step_logps = []
    for pred_now, pred_old in zip(current_preds, trace.old_flow_preds):
        pred_old = pred_old.to(device=pred_now.device, dtype=pred_now.dtype)
        logp_now = _compute_surrogate_step_logprob(pred_now, pred_old, sigma=cfg.surrogate_sigma)
        logp_ref = _compute_surrogate_step_logprob(pred_old, pred_old, sigma=cfg.surrogate_sigma)
        step_logps.append(logp_now)
        ref_step_logps.append(logp_ref)
    logp_now = torch.stack(step_logps, dim=0).sum(dim=0).squeeze(0)
    logp_ref = torch.stack(ref_step_logps, dim=0).sum(dim=0).squeeze(0)
    return logp_now, logp_ref
