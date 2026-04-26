from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from .depth_backends import build_depth_backend
from .idm_backends import build_idm_backend, decode_next_action
from .wan_i2v import WanImageToVideo


def load_config(config_path: str):
    config_file = Path(config_path).resolve()
    repo_root = config_file.parents[2]
    root_cfg = OmegaConf.create({})

    def _merge_if_exists(path: Path):
        nonlocal root_cfg
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                root_cfg = OmegaConf.merge(root_cfg, OmegaConf.create(yaml.safe_load(f)))

    _merge_if_exists(repo_root / "configurations" / "config.yaml")

    experiment_cfg = OmegaConf.create({})
    dataset_cfg = OmegaConf.create({})
    algorithm_cfg = OmegaConf.create({})

    for path in (
        repo_root / "configurations" / "experiment" / "base_experiment.yaml",
        repo_root / "configurations" / "experiment" / "base_pytorch.yaml",
        repo_root / "configurations" / "experiment" / "exp_inference.yaml",
    ):
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                experiment_cfg = OmegaConf.merge(experiment_cfg, OmegaConf.create(yaml.safe_load(f)))

    for path in (
        repo_root / "configurations" / "dataset" / "video_base.yaml",
        repo_root / "configurations" / "dataset" / "image_csv.yaml",
    ):
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                dataset_cfg = OmegaConf.merge(dataset_cfg, OmegaConf.create(yaml.safe_load(f)))

    if config_file.name == "wan_i2v.yaml":
        with (config_file.parent / "wan_t2v.yaml").open("r", encoding="utf-8") as f:
            algorithm_cfg = OmegaConf.merge(algorithm_cfg, OmegaConf.create(yaml.safe_load(f)))
    with config_file.open("r", encoding="utf-8") as f:
        algorithm_cfg = OmegaConf.merge(algorithm_cfg, OmegaConf.create(yaml.safe_load(f)))

    root_cfg.experiment = experiment_cfg
    root_cfg.dataset = dataset_cfg
    root_cfg.algorithm = algorithm_cfg
    root_cfg.algorithm.debug = root_cfg.get("debug", False)

    resolved = OmegaConf.create(OmegaConf.to_container(root_cfg, resolve=True))
    return resolved.algorithm


def build_algo(cfg):
    algo = WanImageToVideo(cfg)
    algo.configure_model()
    algo.eval()
    return algo


def load_condition_bank(path: str):
    bank_path = Path(path)
    if bank_path.suffix == ".pt":
        return torch.load(bank_path)
    raise ValueError(f"Unsupported condition bank format: {bank_path}")


def apply_runtime_overrides(cfg, args):
    with open_dict(cfg):
        if args.override_height is not None:
            cfg.height = int(args.override_height)
        if args.override_width is not None:
            cfg.width = int(args.override_width)
        if args.override_n_frames is not None:
            cfg.n_frames = int(args.override_n_frames)
        if args.override_sample_steps is not None:
            cfg.sample_steps = int(args.override_sample_steps)
        if args.disable_model_compile:
            cfg.model.compile = False
        if cfg.get("clip") is not None and args.disable_clip_compile:
            cfg.clip.compile = False
        if cfg.get("vae") is not None and args.disable_vae_compile:
            cfg.vae.compile = False
        if cfg.get("text_encoder") is not None and args.disable_text_encoder_compile:
            cfg.text_encoder.compile = False
    return cfg


def _resize_video_frames(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    b, t, c, _, _ = video.shape
    flat = video.reshape(b * t, c, video.shape[-2], video.shape[-1])
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(b, t, c, height, width)


def _resize_depth_frames(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    b, t, _, _ = depth.shape
    flat = depth.reshape(b * t, 1, depth.shape[-2], depth.shape[-1])
    resized = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
    return resized.reshape(b, t, height, width)


def _resize_rgb(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(rgb, size=(height, width), mode="bilinear", align_corners=False)


def _resize_depth(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)


def _match_num_frames(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = video.shape[1]
    if current_frames == target_frames:
        return video
    if current_frames > target_frames:
        return video[:, :target_frames]
    pad = video[:, -1:].repeat(1, target_frames - current_frames, *([1] * (video.ndim - 2)))
    return torch.cat([video, pad], dim=1)


def adapt_condition_batch(cond_batch: dict, *, n_frames: int, height: int, width: int) -> dict:
    adapted = {}
    for key, value in cond_batch.items():
        adapted[key] = value.clone() if isinstance(value, torch.Tensor) else value

    if "videos" in adapted:
        videos = adapted["videos"]
        videos = _match_num_frames(videos, n_frames)
        adapted["videos"] = _resize_video_frames(videos, height, width)

    if "target_videos" in adapted:
        target_videos = adapted["target_videos"]
        target_videos = _match_num_frames(target_videos, n_frames)
        adapted["target_videos"] = _resize_video_frames(target_videos, height, width)

    if "depth_video" in adapted:
        depth_video = adapted["depth_video"]
        depth_video = _match_num_frames(depth_video, n_frames)
        adapted["depth_video"] = _resize_depth_frames(depth_video, height, width)

    if "current_rgb" in adapted:
        adapted["current_rgb"] = _resize_rgb(adapted["current_rgb"], height, width)
    elif "target_videos" in adapted:
        adapted["current_rgb"] = adapted["target_videos"][:, 0]

    if "current_depth" in adapted:
        adapted["current_depth"] = _resize_depth(adapted["current_depth"], height, width)
    elif "depth_video" in adapted:
        adapted["current_depth"] = adapted["depth_video"][:, 0]

    if "bbox_render" in adapted:
        adapted["bbox_render"] = _resize_depth_frames(adapted["bbox_render"], height, width)

    return adapted


def get_condition_batch(condition_bank, index: int, device: torch.device, *, n_frames: int, height: int, width: int):
    item = condition_bank[index]
    cond_batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            cond_batch[key] = value.to(device)
        else:
            cond_batch[key] = value
    return adapt_condition_batch(cond_batch, n_frames=n_frames, height=height, width=width)


def select_depth_condition(cond_batch: dict, device: torch.device) -> dict:
    selected = {}
    for key in ("pred_depth_video", "depth_video", "current_depth", "depth"):
        if key in cond_batch:
            value = cond_batch[key]
            selected[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return selected


def select_idm_condition(cond_batch: dict, device: torch.device) -> dict:
    selected = {}
    for key, value in cond_batch.items():
        if isinstance(value, torch.Tensor):
            if key in {"control_state", "current_depth", "pred_depth_video", "depth_video", "left_endpose", "right_endpose", "ee_xyz"}:
                selected[key] = value.to(device)
        else:
            if key in {"prompts", "task_name", "episode_path", "step"}:
                selected[key] = value
    return selected


def main():
    parser = argparse.ArgumentParser(description="Single-step Wan -> depth -> IDM -> next control-state replanning demo.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition-bank", required=True)
    parser.add_argument("--condition-index", type=int, default=0)
    parser.add_argument("--hist-len", type=int, default=1)
    parser.add_argument("--idm-checkpoint", required=True)
    parser.add_argument("--idm-backend", choices=["anypos", "custom"], default="anypos")
    parser.add_argument("--idm-model-name", default="direction_aware_with_split")
    parser.add_argument("--idm-dinov2-name", default="facebook/dinov2-with-registers-base")
    parser.add_argument("--idm-left-arm-dim", type=int, default=6)
    parser.add_argument("--idm-right-arm-dim", type=int, default=6)
    parser.add_argument("--idm-model-output-dim", type=int, default=16)
    parser.add_argument("--idm-custom-backend", default=None)
    parser.add_argument("--depth-backend", choices=["da3", "repeat"], default="da3")
    parser.add_argument("--da3-model-dir", default=None)
    parser.add_argument("--da3-repo-root", default=None)
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-process-res-method", default="upper_bound_resize")
    parser.add_argument("--da3-use-ray-pose", action="store_true")
    parser.add_argument("--da3-ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--wan-device", default=None)
    parser.add_argument("--depth-device", default=None)
    parser.add_argument("--idm-device", default=None)
    parser.add_argument("--override-height", type=int, default=None)
    parser.add_argument("--override-width", type=int, default=None)
    parser.add_argument("--override-n-frames", type=int, default=None)
    parser.add_argument("--override-sample-steps", type=int, default=None)
    parser.add_argument("--disable-model-compile", action="store_true")
    parser.add_argument("--disable-clip-compile", action="store_true")
    parser.add_argument("--disable-vae-compile", action="store_true")
    parser.add_argument("--disable-text-encoder-compile", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_runtime_overrides(cfg, args)
    wan_device = torch.device(args.wan_device if args.wan_device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    depth_device = torch.device(args.depth_device if args.depth_device is not None else wan_device)
    idm_device = torch.device(args.idm_device if args.idm_device is not None else wan_device)
    algo = build_algo(cfg).to(wan_device)
    condition_bank = load_condition_bank(args.condition_bank)
    cond_batch = get_condition_batch(
        condition_bank,
        args.condition_index,
        wan_device,
        n_frames=algo.n_frames,
        height=algo.height,
        width=algo.width,
    )

    idm_backend = build_idm_backend(
        backend_type=args.idm_backend,
        checkpoint=args.idm_checkpoint,
        model_name=args.idm_model_name,
        dinov2_name=args.idm_dinov2_name,
        left_arm_dim=args.idm_left_arm_dim,
        right_arm_dim=args.idm_right_arm_dim,
        model_output_dim=args.idm_model_output_dim,
        custom_backend_target=args.idm_custom_backend,
        device=str(idm_device),
    )
    depth_backend = build_depth_backend(
        backend_type=args.depth_backend,
        model_dir=args.da3_model_dir,
        da3_repo_root=args.da3_repo_root,
        device=str(depth_device),
        process_res=args.da3_process_res,
        process_res_method=args.da3_process_res_method,
        use_ray_pose=args.da3_use_ray_pose,
        ref_view_strategy=args.da3_ref_view_strategy,
    )

    with torch.no_grad():
        predicted_videos = algo.sample_seq(algo.clone_batch(cond_batch), hist_len=args.hist_len)
        next_rgb = predicted_videos[:, args.hist_len: args.hist_len + 1]
        depth_cond = select_depth_condition(cond_batch, depth_device)
        if "current_depth" not in depth_cond and "current_depth" in cond_batch:
            depth_cond["current_depth"] = cond_batch["current_depth"].to(depth_device)
        pred_depth = depth_backend.predict_depth(next_rgb.to(depth_device), depth_cond)
        pred_depth_wan = pred_depth.to(wan_device)
        idm_cond = select_idm_condition(cond_batch, idm_device)
        action_t = decode_next_action(
            idm_backend,
            current_rgb=(cond_batch["current_rgb"].unsqueeze(1) if cond_batch["current_rgb"].ndim == 4 else cond_batch["current_rgb"]).to(idm_device),
            next_rgb=next_rgb.to(idm_device),
            control_state=cond_batch["control_state"].to(idm_device),
            current_depth=(cond_batch["current_depth"].unsqueeze(1) if cond_batch["current_depth"].ndim == 3 else cond_batch["current_depth"]).to(idm_device),
            next_depth=pred_depth,
            cond_batch=idm_cond,
        ).to(wan_device)
    result = {
        "condition_index": args.condition_index,
        "prompt": cond_batch.get("prompts", [""])[0],
        "predicted_action_t": action_t.detach().cpu().tolist(),
        "next_control_state_t1": action_t.detach().cpu().tolist(),
        "predicted_video_shape": list(next_rgb.shape),
        "predicted_depth_shape": list(pred_depth_wan.shape),
        "runtime_cfg": {
            "height": algo.height,
            "width": algo.width,
            "n_frames": algo.n_frames,
            "sample_steps": algo.sample_steps,
            "wan_device": str(wan_device),
            "depth_device": str(depth_device),
            "idm_device": str(idm_device),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
