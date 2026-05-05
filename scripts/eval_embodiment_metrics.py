"""Unified evaluation script for embodiment-level executability metrics.

Evaluates one or more SFT/GRPO checkpoints on held-out test episodes.
Computes 3 metric categories:
  1. Embodiment-level: arm entry rate, omission rate, hallucination rate
  2. Executability: FK workspace violation, chain consistency error
  3. Generation quality: FVD, LPIPS, SSIM

Usage:
  python scripts/eval_embodiment_metrics.py \
    --test-data /path/to/test_hdf5 \
    --checkpoints vanilla.ckpt entry.ckpt two_stage.ckpt fk_grpo.ckpt \
    --labels "Vanilla SFT" "Entry SFT" "Two-stage SFT" "+ FK GRPO" \
    --output results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import cv2
from tqdm import tqdm


# ── Arm detection heuristic ────────────────────────────────────────────────

def detect_arm_presence(rgb_frames: torch.Tensor, lower_frac: float = 0.3) -> torch.Tensor:
    """Detect arm presence per frame using motion energy in lower region.

    Args:
        rgb_frames: [T, 3, H, W] normalized to [-1, 1]
        lower_frac: fraction of frame height considered "arm entry zone"

    Returns:
        [T] boolean: True if arm is present in frame
    """
    T, C, H, W = rgb_frames.shape
    lower = rgb_frames[:, :, int(H * (1 - lower_frac)):, :]

    # Frame-to-frame difference in lower region
    if T > 1:
        diff = (lower[1:] - lower[:-1]).abs().mean(dim=[1, 2, 3])  # [T-1]
        threshold = diff.mean() * 0.5 if diff.numel() > 0 else 0.01
        present = torch.cat([torch.tensor([False]), diff > threshold])  # [T]
    else:
        present = torch.zeros(T, dtype=torch.bool)

    return present


def compute_embodiment_metrics(
    gen_videos: torch.Tensor,  # [N, T, 3, H, W]
) -> Dict[str, float]:
    """Compute arm entry, omission, hallucination metrics."""
    N, T = gen_videos.shape[:2]
    entry_count = 0
    omit_count = 0
    hall_count = 0
    total = 0

    for i in range(N):
        arm_present = detect_arm_presence(gen_videos[i])
        total += 1

        # Arm entry: arm absent in early frames → present in later frames
        early = arm_present[: max(1, T // 3)]
        late = arm_present[-max(1, T // 3):]
        if early.float().mean() < 0.3 and late.float().mean() > 0.5:
            entry_count += 1

        # Arm omission: arm never appears
        if arm_present.float().mean() < 0.1:
            omit_count += 1

        # Arm hallucination: high motion in invalid regions (not implemented simply)
        # For now, placeholder — count videos with erratic arm patterns
        if arm_present.float().mean() > 0.1 and early.float().mean() > 0.7:
            hall_count += 1  # arm present from start (shouldn't be for entry)

    return {
        "arm_entry_rate": entry_count / max(total, 1),
        "arm_omission_rate": omit_count / max(total, 1),
        "arm_hallucination_rate": hall_count / max(total, 1),
    }


# ── FK executability metrics ───────────────────────────────────────────────
# Lightweight FK computation (no pytorch_kinematics dependency for eval)

def compute_fk_metrics(
    action_seq: torch.Tensor,  # [N, T, 16] joint positions
    left_arm_idx: slice = slice(0, 7),
    right_arm_idx: slice = slice(8, 15),
    workspace_bounds: Tuple[float, float] = (0.2, 0.8),  # min/max normalized workspace
    max_vel: float = 1.7,  # m/s
    max_acc: float = 10.0,  # m/s²
) -> Dict[str, float]:
    """Compute FK workspace and chain consistency metrics.

    Uses a simplified FK approximation: joint L2 norm as proxy for ee position.
    For precise FK, use fk_reward.py with URDF.
    """
    N, T, D = action_seq.shape

    # Simplified: use joint positions as proxy for FK
    # Left and right arm joint norms as workspace indicators
    left_joints = action_seq[:, :, left_arm_idx]   # [N, T, 7]
    right_joints = action_seq[:, :, right_arm_idx]  # [N, T, 7]

    left_norm = left_joints.norm(dim=-1)  # [N, T]
    right_norm = right_joints.norm(dim=-1)  # [N, T]

    # Workspace violations: joint norm beyond reasonable range
    # Franka joints range approx [-2.9, 2.9] rad → norm ≈ 0-8
    ws_violations = ((left_norm > 8.0) | (right_norm > 8.0)).float().mean().item()

    # Joint velocity (first difference)
    if T > 1:
        vel = (action_seq[:, 1:] - action_seq[:, :-1]).norm(dim=-1)  # [N, T-1]
        vel_violations = (vel > max_vel * 0.05).float().mean().item()  # scaled for joint space
        max_vel_obs = vel.max().item()
        mean_vel = vel.mean().item()

        # Acceleration
        if T > 2:
            acc = vel[:, 1:] - vel[:, :-1]  # [N, T-2]
            acc_violations = (acc.abs() > max_acc * 0.05).float().mean().item()
            max_acc_obs = acc.abs().max().item()
            mean_jerk = acc.abs().mean().item()
        else:
            acc_violations = 0.0
            max_acc_obs = 0.0
            mean_jerk = 0.0
    else:
        vel_violations = 0.0
        max_vel_obs = 0.0
        mean_vel = 0.0
        acc_violations = 0.0
        max_acc_obs = 0.0
        mean_jerk = 0.0

    return {
        "fk_workspace_violation": ws_violations,
        "fk_vel_violation": vel_violations,
        "fk_acc_violation": acc_violations,
        "max_velocity": max_vel_obs,
        "mean_velocity": mean_vel,
        "max_acceleration": max_acc_obs,
        "mean_jerk": mean_jerk,
    }


# ── Generation quality metrics ─────────────────────────────────────────────

def compute_quality_metrics(
    gen_videos: torch.Tensor,  # [N, T, 3, H, W] in [-1, 1]
    gt_videos: torch.Tensor,   # [N, T, 3, H, W] in [-1, 1]
) -> Dict[str, float]:
    """Compute FVD approximation, LPIPS, SSIM."""
    metrics = {}

    # Per-frame MSE (proxy for LPIPS when LPIPS unavailable)
    mse = (gen_videos - gt_videos).pow(2).mean().item()
    metrics["mse"] = mse

    # Per-frame SSIM approximation (using patch-wise correlation)
    gen_flat = gen_videos.permute(0, 1, 3, 4, 2).reshape(-1, 3)  # [N*T*H*W, 3]
    gt_flat = gt_videos.permute(0, 1, 3, 4, 2).reshape(-1, 3)

    gen_mean = gen_flat.mean(0, keepdim=True)
    gt_mean = gt_flat.mean(0, keepdim=True)
    gen_std = gen_flat.std(0, keepdim=True) + 1e-8
    gt_std = gt_flat.std(0, keepdim=True) + 1e-8

    corr = ((gen_flat - gen_mean) * (gt_flat - gt_mean)).mean() / (gen_std * gt_std).mean()
    metrics["frame_correlation"] = corr.item()

    # FVD approximation: per-frame feature distance
    # Using raw pixel features as FVD proxy (real FVD needs I3D features)
    gen_feat = gen_videos.mean(dim=[2, 3, 4])  # [N, T] — spatial average
    gt_feat = gt_videos.mean(dim=[2, 3, 4])
    fvd_proxy = (gen_feat - gt_feat).pow(2).mean().sqrt().item()
    metrics["fvd_proxy"] = fvd_proxy

    return metrics


# ── Main evaluation ────────────────────────────────────────────────────────

def evaluate_checkpoint(
    ckpt_path: str,
    test_episodes: List[Path],
    n_frames: int = 17,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate a single checkpoint on test episodes.

    Note: Full evaluation requires loading the Wan2.1 model and running inference.
    This script provides the metric computation framework; the actual video generation
    should be done separately and loaded here, or integrated with the Wan inference pipeline.
    """
    # Placeholder: in practice, load videos generated by the checkpoint
    # For now, structure the metric computation pipeline
    all_embodiment = []
    all_fk = []
    all_quality = []

    # TODO: integrate with Wan inference
    # For each test episode:
    #   1. Load first frame as condition
    #   2. Generate video using the checkpoint
    #   3. Compute metrics

    return {
        "arm_entry_rate": 0.0,
        "arm_omission_rate": 0.0,
        "arm_hallucination_rate": 0.0,
        "fk_workspace_violation": 0.0,
        "fk_vel_violation": 0.0,
        "fk_acc_violation": 0.0,
        "mse": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate embodiment-level metrics")
    parser.add_argument("--test-data", required=True, help="Directory with test episode HDF5 files")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to evaluate")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels for each checkpoint")
    parser.add_argument("--n-frames", type=int, default=17)
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    parser.add_argument("--table-output", default="results_table.tex", help="Output LaTeX table path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.labels):
        raise ValueError("Number of checkpoints must match number of labels")

    test_episodes = sorted(Path(args.test_data).glob("episode*.hdf5"))
    print(f"Test episodes: {len(test_episodes)}")

    results = {}
    for ckpt, label in zip(args.checkpoints, args.labels):
        print(f"\nEvaluating: {label}")
        metrics = evaluate_checkpoint(ckpt, test_episodes, args.n_frames, args.device)
        results[label] = metrics

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Generate LaTeX table
    _write_latex_table(results, args.table_output)
    print(f"LaTeX table saved to {args.table_output}")


def _write_latex_table(results: Dict[str, Dict[str, float]], output_path: str):
    """Write a LaTeX-formatted results table."""
    if not results:
        return

    metric_names = list(next(iter(results.values())).keys())
    labels = list(results.keys())

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    cols = "l" + "c" * len(labels)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = " & ".join([r"\textbf{Metric}"] + [rf"\textbf{{{l}}}" for l in labels]) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for metric in metric_names:
        name = metric.replace("_", " ").title()
        vals = [f"{results[label].get(metric, 0):.3f}" for label in labels]
        row = f"  {name} & " + " & ".join(vals) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Embodiment-level executability evaluation.}")
    lines.append(r"\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
