from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DA3_ROOT = WORKSPACE_ROOT / "Depth-Anything-3-main"


class DepthBackend(Protocol):
    def predict_depth(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            videos: [B, T, C, H, W]
        Returns:
            depth: [B, T, H, W]
        """


def _resize_depth_to_video(depth: torch.Tensor, videos: torch.Tensor) -> torch.Tensor:
    if depth.shape[-2:] == videos.shape[-2:]:
        return depth
    return F.interpolate(
        depth.view(-1, 1, depth.shape[-2], depth.shape[-1]),
        size=videos.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).view(depth.shape[0], depth.shape[1], videos.shape[-2], videos.shape[-1])


class RepeatCurrentDepthBackend:
    """
    Minimal depth completion backend for bring-up.

    Priority:
    1. `cond_batch["pred_depth_video"]`
    2. `cond_batch["depth_video"]`
    3. `cond_batch["current_depth"]` / `cond_batch["depth"]`, repeated over time
    """

    def predict_depth(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        if "pred_depth_video" in cond_batch:
            depth = cond_batch["pred_depth_video"].to(videos.device, dtype=torch.float32)
        elif "depth_video" in cond_batch:
            depth = cond_batch["depth_video"].to(videos.device, dtype=torch.float32)
        elif "current_depth" in cond_batch:
            current = cond_batch["current_depth"].to(videos.device, dtype=torch.float32)
            depth = current.unsqueeze(1).repeat(1, videos.shape[1], 1, 1)
        elif "depth" in cond_batch:
            current = cond_batch["depth"].to(videos.device, dtype=torch.float32)
            depth = current.unsqueeze(1).repeat(1, videos.shape[1], 1, 1)
        else:
            raise ValueError(
                "Depth backend could not find any of pred_depth_video / depth_video / current_depth / depth in cond_batch."
            )

        return _resize_depth_to_video(depth, videos)


class DepthAnything3Backend:
    """
    Real depth backend using Depth Anything 3.

    This backend accepts normalized videos `[B, T, C, H, W]` in `[-1, 1]`, runs
    DA3 on each sample sequence, and returns `[B, T, H, W]` float32 depth maps.
    """

    def __init__(
        self,
        *,
        model_dir: str,
        repo_root: str | None = None,
        device: str | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ):
        active_repo_root = Path(repo_root) if repo_root is not None else DEFAULT_DA3_ROOT
        if not active_repo_root.exists():
            raise FileNotFoundError(f"Depth Anything 3 repo root not found: {active_repo_root}")

        src_root = active_repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        if str(active_repo_root) not in sys.path:
            sys.path.insert(0, str(active_repo_root))

        from depth_anything_3.api import DepthAnything3

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_dir = model_dir
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.use_ray_pose = bool(use_ray_pose)
        self.ref_view_strategy = ref_view_strategy

        self.model = DepthAnything3.from_pretrained(model_dir)
        self.model = self.model.to(device=self.device)
        self.model.eval()

    @staticmethod
    def _video_to_numpy_list(video: torch.Tensor) -> list[np.ndarray]:
        frames = (
            video.detach()
            .float()
            .clamp(-1, 1)
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        return [np.ascontiguousarray(frame) for frame in frames]

    @torch.inference_mode()
    def predict_depth(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        if "pred_depth_video" in cond_batch:
            depth = cond_batch["pred_depth_video"].to(videos.device, dtype=torch.float32)
            return _resize_depth_to_video(depth, videos)

        if videos.ndim != 5:
            raise ValueError(f"Expected videos [B, T, C, H, W], got {tuple(videos.shape)}")

        depth_batches = []
        for video in videos:
            prediction = self.model.inference(
                self._video_to_numpy_list(video),
                process_res=self.process_res,
                process_res_method=self.process_res_method,
                use_ray_pose=self.use_ray_pose,
                ref_view_strategy=self.ref_view_strategy,
                export_dir=None,
            )
            depth = torch.from_numpy(np.asarray(prediction.depth, dtype=np.float32))
            if depth.ndim != 3:
                raise ValueError(f"Expected DA3 depth output [T, H, W], got {tuple(depth.shape)}")
            depth_batches.append(depth)

        stacked = torch.stack(depth_batches, dim=0).to(videos.device, dtype=torch.float32)
        return _resize_depth_to_video(stacked, videos)


def build_depth_backend(
    *,
    backend_type: str,
    model_dir: str | None = None,
    da3_repo_root: str | None = None,
    device: str | None = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
) -> DepthBackend:
    backend_type = backend_type.lower()
    if backend_type == "repeat":
        return RepeatCurrentDepthBackend()
    if backend_type == "da3":
        if model_dir is None:
            raise ValueError("model_dir is required for backend_type=da3")
        return DepthAnything3Backend(
            model_dir=model_dir,
            repo_root=da3_repo_root,
            device=device,
            process_res=process_res,
            process_res_method=process_res_method,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
        )
    raise ValueError(f"Unsupported depth backend type: {backend_type}")
