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

    Preferred interface: backends implement `decode_next_action(...) -> [B, D]`.

    Fallback for sequence decoders: concatenate current and next RGB frames,
    run `decode_actions`, and return the final decoded action.
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


# ---------------------------------------------------------------------------
# Vidar IDM backend
# ---------------------------------------------------------------------------

VIDAR_ROOT = WORKSPACE_ROOT / "vidar"
if str(VIDAR_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDAR_ROOT))


@dataclass
class VidarBackendConfig:
    checkpoint: str
    model_name: str = "mask"
    output_dim: int = 16
    device: str | None = None


class _VidarPreprocessorArgs:
    use_transform = False


class VidarIDMBackend:
    """
    Bridge adapter for using the Vidar dual-frame Masked-IDM as the IDM backend.

    The Vidar MaskedIDM takes:
        img_t      [B, 3, H, W]   RGB frame at time t   (ImageNet-normalised)
        dep_t      [B, 1, H, W]   depth at time t       (normalised to [-1, 1])
        img_next   [B, 3, H, W]   RGB frame at time t+1
        dep_next   [B, 1, H, W]   depth at time t+1
        pos_t      [B, 16]        current joint state
    and directly outputs:
        pos_{t+1}  [B, 16]        next joint state

    No action adapter is needed because the model already outputs 16-dim
    Franka control states.
    """

    def __init__(self, cfg: VidarBackendConfig):
        from idm.idm import IDM as VidarIDMModel  # type: ignore  # noqa: E402
        from idm.preprocessor import DinoPreprocessor as VidarPreprocessor  # type: ignore  # noqa: E402

        self.cfg = cfg
        self.device = torch.device(
            cfg.device if cfg.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_dim = int(cfg.output_dim)  # 16

        # Build model
        self.model = VidarIDMModel(model_name=cfg.model_name, output_dim=cfg.output_dim)

        # Load checkpoint
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

        # Preprocessor (inference mode, no augmentation)
        self.preprocessor = VidarPreprocessor(_VidarPreprocessorArgs())

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _video_to_rgb01(self, videos: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] video tensor to [0, 1]."""
        if videos.ndim != 5:
            raise ValueError(f"Expected videos [B, T, C, H, W], got {tuple(videos.shape)}")
        return videos.float().clamp(-1, 1).add(1.0).mul_(0.5)

    def _preprocess_rgb(self, rgb_01: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0,1] -> ImageNet-normalised, resized to 518x518."""
        images_np = (
            rgb_01.mul(255.0)
            .round()
            .clamp(0, 255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        processed = [self.preprocessor.process_image(img) for img in images_np]
        return torch.stack(processed, dim=0).to(self.device)

    def _preprocess_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """[B, H, W] or [B, 1, H, W] depth in arbitrary range -> normalised [B, 1, 518, 518]."""
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)  # [B, 1, H, W]
        # Resize to 518x518
        resized = F.interpolate(depth.float(), size=(518, 518), mode="bilinear", align_corners=False)
        # Normalise to [-1, 1] (same convention as vidar preprocessor)
        resized = (resized - resized.mean()) / (resized.std() + 1e-6)
        resized = resized.clamp(-3, 3) / 3.0  # soft clamp to [-1, 1]
        return resized.to(self.device)

    # ------------------------------------------------------------------
    # Core decode methods
    # ------------------------------------------------------------------

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
        """
        Decode a single-step action for transition t -> t+1.

        Returns:
            action_t: [B, 16] — the predicted next joint state (pos_{t+1}).
        """
        # current_rgb / next_rgb: [B, 1, C, H, W] or [B, C, H, W]
        if current_rgb.ndim == 5:
            current_rgb = current_rgb[:, 0]  # [B, C, H, W]
        if next_rgb.ndim == 5:
            next_rgb = next_rgb[:, 0]

        # Convert from [-1,1] to [0,1]
        cur_rgb_01 = current_rgb.float().clamp(-1, 1).add(1.0).mul_(0.5)
        nxt_rgb_01 = next_rgb.float().clamp(-1, 1).add(1.0).mul_(0.5)

        # Preprocess RGB (ImageNet normalisation + resize to 518)
        img_t = self._preprocess_rgb(cur_rgb_01)
        img_next = self._preprocess_rgb(nxt_rgb_01)

        # Preprocess depth
        if current_depth is not None:
            if current_depth.ndim == 4 and current_depth.shape[1] > 1:
                current_depth = current_depth[:, 0]  # take first frame
            dep_t = self._preprocess_depth(current_depth)
        else:
            dep_t = torch.zeros(img_t.shape[0], 1, 518, 518, device=self.device)

        if next_depth is not None:
            if next_depth.ndim == 4 and next_depth.shape[1] > 1:
                next_depth = next_depth[:, 0]
            dep_next = self._preprocess_depth(next_depth)
        else:
            dep_next = torch.zeros(img_t.shape[0], 1, 518, 518, device=self.device)

        # Control state
        if control_state is None:
            pos_t = torch.zeros(img_t.shape[0], 16, device=self.device)
        else:
            pos_t = control_state.to(device=self.device, dtype=torch.float32)
            if pos_t.ndim == 1:
                pos_t = pos_t.unsqueeze(0)

        # Forward pass
        action_t = self.model(img_t, dep_t, img_next, dep_next, pos_t)
        return action_t  # [B, 16]

    @torch.no_grad()
    def decode_actions(self, videos: torch.Tensor, cond_batch: dict[str, Any]) -> torch.Tensor:
        """
        Decode actions for a video sequence by processing consecutive frame pairs.

        Args:
            videos: [B, T, C, H, W] in [-1, 1]
            cond_batch: must contain 'control_state' [B, 16]

        Returns:
            actions: [B, T-1, 16] — predicted next states for each transition.
        """
        rgb01 = self._video_to_rgb01(videos)
        batch_size, horizon = rgb01.shape[:2]
        if horizon < 2:
            raise ValueError(f"Need at least 2 frames for decode_actions, got {horizon}")

        control_state = cond_batch.get("control_state")
        depth_video = cond_batch.get("pred_depth_video", cond_batch.get("depth_video"))

        actions = []
        current_state = control_state
        for t in range(horizon - 1):
            cur_rgb = rgb01[:, t]  # [B, C, H, W] in [0,1]
            nxt_rgb = rgb01[:, t + 1]

            cur_dep = depth_video[:, t] if depth_video is not None else None
            nxt_dep = depth_video[:, t + 1] if depth_video is not None else None

            img_t = self._preprocess_rgb(cur_rgb)
            img_next = self._preprocess_rgb(nxt_rgb)
            dep_t = self._preprocess_depth(cur_dep) if cur_dep is not None else torch.zeros(batch_size, 1, 518, 518, device=self.device)
            dep_next = self._preprocess_depth(nxt_dep) if nxt_dep is not None else torch.zeros(batch_size, 1, 518, 518, device=self.device)

            if current_state is None:
                pos_t = torch.zeros(batch_size, 16, device=self.device)
            else:
                pos_t = current_state.to(device=self.device, dtype=torch.float32)
                if pos_t.ndim == 1:
                    pos_t = pos_t.unsqueeze(0)

            action_t = self.model(img_t, dep_t, img_next, dep_next, pos_t)
            actions.append(action_t)
            current_state = action_t  # autoregressive: predicted state becomes next input

        return torch.stack(actions, dim=1)  # [B, T-1, 16]


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
    model_name: str = "mask",
    dinov2_name: str = "",
    freeze_dinov2: bool = False,
    left_arm_dim: int = 7,
    right_arm_dim: int = 7,
    model_output_dim: int | None = None,
    target_action_dim: int | None = 16,
    action_adapter: str = "identity",
    custom_backend_target: str | None = None,
    custom_backend_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
) -> IDMBackend:
    backend_type = backend_type.lower()
    if backend_type == "vidar":
        if checkpoint is None:
            raise ValueError("checkpoint is required for backend_type=vidar")
        return VidarIDMBackend(
            VidarBackendConfig(
                checkpoint=checkpoint,
                device=device,
            )
        )
    if backend_type == "custom":
        if custom_backend_target is None:
            raise ValueError("custom_backend_target is required for backend_type=custom")
        return CustomIDMBackend(custom_backend_target, custom_backend_kwargs)
    raise ValueError(f"Unsupported IDM backend type: {backend_type}. Use 'vidar' or 'custom'.")
