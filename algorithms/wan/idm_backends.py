from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ANYPOS_ROOT = WORKSPACE_ROOT / "AnyPos-main"

if str(ANYPOS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYPOS_ROOT))

from idm.idm import IDM as AnyPosIDMModel  # type: ignore  # noqa: E402
from idm.preprocessor import DinoPreprocessor, segment_robot_arms  # type: ignore  # noqa: E402

try:
    from idm.spec import infer_action_dim  # type: ignore  # noqa: E402
except Exception:
    def infer_action_dim(output_dim: int | None, left_arm_dim: int, right_arm_dim: int) -> int:
        if output_dim is not None:
            return int(output_dim)
        return int(left_arm_dim + 1 + right_arm_dim + 1)


class IDMBackend(Protocol):
    action_dim: int

    def decode_actions(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            videos: [B, T, C, H, W] in [-1, 1]
        Returns:
            actions: [B, T, D]
        """


def decode_next_action(
    idm_backend: IDMBackend,
    *,
    current_rgb: torch.Tensor,
    next_rgb: torch.Tensor,
    control_state: torch.Tensor | None,
    current_depth: torch.Tensor | None = None,
    next_depth: torch.Tensor | None = None,
    cond_batch: dict[str, Any] | None = None,
) -> torch.Tensor:
    """
    Decode a single-step next action for the transition `t -> t+1`.

    Preferred future interface:
    custom IDM backends can implement `decode_next_action(...) -> [B, D]`.

    Current fallback for AnyPos-style sequence decoders:
    concatenate the current and next RGB frames, optionally attach a 2-frame
    depth video, run `decode_actions`, and use the final decoded action as
    `action_t`, which aligns with `action[t] = control_state[t+1]`.
    """
    active_cond = dict(cond_batch or {})
    if control_state is not None:
        active_cond["control_state"] = control_state
    if current_depth is not None:
        active_cond["current_depth"] = current_depth[:, 0] if current_depth.ndim == 4 else current_depth
    if current_depth is not None and next_depth is not None:
        active_cond["pred_depth_video"] = torch.cat([current_depth, next_depth], dim=1)

    backend_impl = getattr(idm_backend, "backend", idm_backend)
    if hasattr(backend_impl, "decode_next_action"):
        return backend_impl.decode_next_action(
            current_rgb=current_rgb,
            next_rgb=next_rgb,
            control_state=control_state,
            current_depth=current_depth,
            next_depth=next_depth,
            cond_batch=active_cond,
        )

    transition_video = torch.cat([current_rgb, next_rgb], dim=1)
    decoded = idm_backend.decode_actions(transition_video, active_cond)
    if decoded.ndim != 3:
        raise ValueError(f"Expected decoded actions [B, T, D], got {tuple(decoded.shape)}")
    return decoded[:, -1]


@dataclass
class AnyPosBackendConfig:
    checkpoint: str
    model_name: str = "direction_aware_with_split"
    dinov2_name: str = "facebook/dinov2-with-registers-base"
    freeze_dinov2: bool = False
    left_arm_dim: int = 6
    right_arm_dim: int = 6
    model_output_dim: int | None = None
    device: str | None = None


class _AnyPosPreprocessorArgs:
    use_transform = False


class AnyPosIDMBackend:
    """
    Bridge adapter for using AnyPos as a temporary IDM backend.

    Important limitation:
    AnyPos is an RGB-only image-to-action model. It does not match the target
    RGB-D_t + control-state_t + RGB-D_{t+1} -> action_t formulation exactly.
    We therefore use it here only as a bring-up backend to unblock:
    - GRPO reward wiring
    - online replan plumbing
    - future backend swap to the project's own IDM
    """

    def __init__(self, cfg: AnyPosBackendConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_output_dim = infer_action_dim(cfg.model_output_dim, cfg.left_arm_dim, cfg.right_arm_dim)
        self.action_dim = int(self.model_output_dim)
        self.preprocessor = DinoPreprocessor(_AnyPosPreprocessorArgs())
        model_kwargs = dict(
            model_name=cfg.model_name,
            dinov2_name=cfg.dinov2_name,
            freeze_dinov2=cfg.freeze_dinov2,
            output_dim=self.model_output_dim,
        )
        try:
            self.model = AnyPosIDMModel(
                **model_kwargs,
                left_arm_dim=cfg.left_arm_dim,
                right_arm_dim=cfg.right_arm_dim,
            )
        except TypeError:
            self.model = AnyPosIDMModel(**model_kwargs)
        try:
            checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _video_to_rgb01(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f"Expected videos [B, T, C, H, W], got {tuple(videos.shape)}")
        return videos.float().clamp(-1, 1).add(1.0).mul_(0.5)

    def _segment_regions(self, image: np.ndarray) -> torch.Tensor:
        _, arm_boxes = segment_robot_arms(image)
        h, w = image.shape[:2]
        left_split = int((arm_boxes["left_split"] / w) * 518)
        right_split = int((arm_boxes["right_split"] / w) * 518)
        arm_split = int((arm_boxes["arm_gripper_split"] / h) * 518)
        gripper_split = int((arm_boxes["gripper_split"] / w) * 518)

        processed = self.preprocessor.process_image(image)
        regions = []
        for region_idx in range(4):
            if region_idx == 0:
                region = processed[:, :arm_split, :left_split]
            elif region_idx == 1:
                region = processed[:, arm_split:, :gripper_split]
            elif region_idx == 2:
                region = processed[:, :arm_split, right_split:]
            else:
                region = processed[:, arm_split:, gripper_split:]
            resized = F.interpolate(
                region.unsqueeze(0),
                size=(processed.shape[1], processed.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            regions.append(resized)
        return torch.stack(regions, dim=0)

    def _prepare_batch(self, frames_rgb01: torch.Tensor) -> torch.Tensor:
        images_np = (
            frames_rgb01.mul(255.0)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        if "with_split" in self.cfg.model_name:
            region_batches = [self._segment_regions(image) for image in images_np]
            stacked = torch.stack(region_batches, dim=1)
            return stacked.to(self.device)
        processed = [self.preprocessor.process_image(image) for image in images_np]
        return torch.stack(processed, dim=0).to(self.device)

    @torch.no_grad()
    def decode_actions(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        del cond_batch
        rgb01 = self._video_to_rgb01(videos)
        batch_size, horizon = rgb01.shape[:2]
        flat_frames = rgb01.reshape(batch_size * horizon, *rgb01.shape[2:])
        model_inputs = self._prepare_batch(flat_frames)
        predictions = self.model(model_inputs).view(batch_size, horizon, -1)
        if predictions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected native IDM output dim {self.action_dim}, got {predictions.shape[-1]}")
        return predictions

    @torch.no_grad()
    def decode_next_action(
        self,
        *,
        current_rgb: torch.Tensor,
        next_rgb: torch.Tensor,
        control_state: torch.Tensor | None,
        current_depth: torch.Tensor | None = None,
        next_depth: torch.Tensor | None = None,
        cond_batch: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        del current_depth, next_depth, control_state, cond_batch
        transition_video = torch.cat([current_rgb, next_rgb], dim=1)
        rgb01 = self._video_to_rgb01(transition_video)
        batch_size = rgb01.shape[0]
        flat_frames = rgb01.reshape(batch_size * rgb01.shape[1], *rgb01.shape[2:])
        model_inputs = self._prepare_batch(flat_frames)
        predictions = self.model(model_inputs).view(batch_size, rgb01.shape[1], -1)
        if predictions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected native IDM output dim {self.action_dim}, got {predictions.shape[-1]}")
        return predictions[:, -1]


class CustomIDMBackend:
    """
    Load a future in-house IDM backend without changing the GRPO or planning loops.

    The target class must expose either:
    - `decode_actions(videos, cond_batch) -> [B, T, D]`
    - or `__call__(videos, cond_batch) -> [B, T, D]`
    """

    def __init__(self, target: str, kwargs: dict[str, Any] | None = None):
        module_name, class_name = target.split(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self.backend = cls(**(kwargs or {}))
        self.action_dim = int(getattr(self.backend, "action_dim"))

    def decode_actions(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        if hasattr(self.backend, "decode_actions"):
            return self.backend.decode_actions(videos, cond_batch)
        return self.backend(videos, cond_batch)


def build_idm_backend(
    *,
    backend_type: str,
    checkpoint: str | None = None,
    model_name: str = "direction_aware_with_split",
    dinov2_name: str = "facebook/dinov2-with-registers-base",
    freeze_dinov2: bool = False,
    left_arm_dim: int = 6,
    right_arm_dim: int = 6,
    model_output_dim: int | None = None,
    custom_backend_target: str | None = None,
    custom_backend_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
) -> IDMBackend:
    backend_type = backend_type.lower()
    if backend_type == "anypos":
        if checkpoint is None:
            raise ValueError("checkpoint is required for backend_type=anypos")
        return AnyPosIDMBackend(
            AnyPosBackendConfig(
                checkpoint=checkpoint,
                model_name=model_name,
                dinov2_name=dinov2_name,
                freeze_dinov2=freeze_dinov2,
                left_arm_dim=left_arm_dim,
                right_arm_dim=right_arm_dim,
                model_output_dim=model_output_dim,
                device=device,
            )
        )
    if backend_type == "custom":
        if custom_backend_target is None:
            raise ValueError("custom_backend_target is required for backend_type=custom")
        return CustomIDMBackend(custom_backend_target, custom_backend_kwargs)
    raise ValueError(f"Unsupported IDM backend type: {backend_type}")
