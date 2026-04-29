from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


def decode_rgb_frame(encoded_frame) -> np.ndarray:
    if isinstance(encoded_frame, np.ndarray) and encoded_frame.dtype == np.uint8 and encoded_frame.ndim == 3:
        image = encoded_frame
    else:
        image = cv2.imdecode(np.frombuffer(encoded_frame, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RGB frame.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_depth(depth_frame) -> np.ndarray:
    depth = np.asarray(depth_frame, dtype=np.float32)
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return np.zeros_like(depth, dtype=np.float32)
    valid = depth[finite_mask]
    low = float(np.quantile(valid, 0.02))
    high = float(np.quantile(valid, 0.98))
    if high <= low:
        high = low + 1e-6
    depth = np.clip(depth, low, high)
    depth = (depth - low) / (high - low)
    depth[~finite_mask] = 0.0
    return depth.astype(np.float32)


def load_window(
    episode_path: Path,
    *,
    step: int,
    n_frames: int,
    image_size: tuple[int, int],
    camera_name: str,
) -> dict:
    with h5py.File(episode_path, "r") as root:
        rgb_ds = root[f"/observation/{camera_name}/rgb"]
        depth_ds = root[f"/observation/{camera_name}/depth"]
        joint_action = np.asarray(root["/joint_action/vector"], dtype=np.float32)
        left_endpose = np.asarray(root["/endpose/left_endpose"], dtype=np.float32) if "/endpose/left_endpose" in root else None
        right_endpose = np.asarray(root["/endpose/right_endpose"], dtype=np.float32) if "/endpose/right_endpose" in root else None
        left_gripper = np.asarray(root["/endpose/left_gripper"], dtype=np.float32) if "/endpose/left_gripper" in root else None
        right_gripper = np.asarray(root["/endpose/right_gripper"], dtype=np.float32) if "/endpose/right_gripper" in root else None

        max_step = min(len(rgb_ds), len(depth_ds), joint_action.shape[0])
        if step >= max_step:
            raise ValueError(f"Window step {step} exceeds episode length {max_step}.")
        window_end = min(step + n_frames, max_step)
        actual_frames = window_end - step

        rgb_frames = []
        depth_frames = []
        for frame_idx in range(step, window_end):
            rgb = cv2.resize(
                decode_rgb_frame(rgb_ds[frame_idx]),
                image_size,
                interpolation=cv2.INTER_LINEAR,
            )
            depth = cv2.resize(
                normalize_depth(depth_ds[frame_idx]),
                image_size,
                interpolation=cv2.INTER_LINEAR,
            )
            rgb_frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
            depth_frames.append(torch.from_numpy(depth).float())

        target_videos = torch.stack(rgb_frames, dim=0).mul_(2.0).sub_(1.0)
        cond_videos = target_videos[:1].repeat(n_frames, 1, 1, 1)
        depth_video = torch.stack(depth_frames, dim=0)
        control_state = torch.from_numpy(joint_action[step]).float()
        valid_horizon = max(min(actual_frames - 1, n_frames - 1), 0)
        future_actions = torch.from_numpy(joint_action[step + 1: step + 1 + valid_horizon]).float()
        action_seq_mask = torch.zeros(n_frames - 1, dtype=torch.float32)
        if valid_horizon > 0:
            action_seq_mask[:valid_horizon] = 1.0
        if future_actions.shape[0] == 0:
            future_actions = control_state.unsqueeze(0)
        if future_actions.shape[0] < n_frames - 1:
            pad = future_actions[-1:].repeat(n_frames - 1 - future_actions.shape[0], 1)
            future_actions = torch.cat([future_actions, pad], dim=0)
        action = future_actions[0]
        left_endpose_t = torch.from_numpy(left_endpose[step]).float() if left_endpose is not None else None
        right_endpose_t = torch.from_numpy(right_endpose[step]).float() if right_endpose is not None else None

    height, width = image_size[1], image_size[0]
    item = {
        "videos": cond_videos.unsqueeze(0),
        "target_videos": target_videos.unsqueeze(0),
        "depth_video": depth_video.unsqueeze(0),
        "current_depth": depth_video[0].unsqueeze(0),
        "current_rgb": target_videos[0].unsqueeze(0),
        "control_state": control_state.unsqueeze(0),
        "action": action.unsqueeze(0),
        "action_seq": future_actions.unsqueeze(0),
        "action_seq_mask": action_seq_mask.unsqueeze(0),
        "valid_horizon": torch.tensor([valid_horizon], dtype=torch.long),
        "actual_frames": torch.tensor([actual_frames], dtype=torch.long),
        "prompts": [episode_path.parent.parent.name or "robot_task"],
        "task_name": episode_path.parent.parent.name or "robot_task",
        "episode_path": str(episode_path),
        "step": int(step),
        "has_bbox": torch.zeros(1, 2, dtype=torch.float32),
        "bbox_render": torch.zeros(1, 2, height, width, dtype=torch.float32),
    }
    if left_endpose_t is not None and right_endpose_t is not None:
        item["left_endpose"] = left_endpose_t.unsqueeze(0)
        item["right_endpose"] = right_endpose_t.unsqueeze(0)
        item["ee_xyz"] = torch.stack([left_endpose_t[:3], right_endpose_t[:3]], dim=0).unsqueeze(0)
    if left_gripper is not None and right_gripper is not None:
        item["left_gripper"] = torch.tensor([left_gripper[step]], dtype=torch.float32)
        item["right_gripper"] = torch.tensor([right_gripper[step]], dtype=torch.float32)
    return item


def list_episode_paths(data_dir: Path, limit: int | None) -> list[Path]:
    episodes = sorted(data_dir.glob("episode*.hdf5"))
    if limit is not None:
        episodes = episodes[:limit]
    if not episodes:
        raise FileNotFoundError(f"No episode*.hdf5 files found under {data_dir}")
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Build a RoboTwin -> Wan state-unrolled GRPO condition bank.")
    parser.add_argument("--data-dir", required=True, help="Directory containing episode*.hdf5")
    parser.add_argument("--output", required=True, help="Output .pt path")
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--n-frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--min-valid-horizon",
        type=int,
        default=1,
        help="Minimum number of real future action labels required to keep a window.",
    )
    parser.add_argument("--index-json", default=None, help="Optional sidecar json with item metadata")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episodes = list_episode_paths(data_dir, args.episode_limit)
    image_size = (args.width, args.height)

    bank = []
    index_records = []
    for episode_path in episodes:
        with h5py.File(episode_path, "r") as root:
            episode_len = min(
                len(root[f"/observation/{args.camera_name}/rgb"]),
                len(root[f"/observation/{args.camera_name}/depth"]),
                root["/joint_action/vector"].shape[0],
            )
        for step in range(0, episode_len, args.step_stride):
            item = load_window(
                episode_path,
                step=step,
                n_frames=args.n_frames,
                image_size=image_size,
                camera_name=args.camera_name,
            )
            if int(item["valid_horizon"].item()) < args.min_valid_horizon:
                continue
            bank.append(item)
            index_records.append(
                {
                    "episode_path": str(episode_path),
                    "step": step,
                    "prompt": item["prompts"][0],
                    "valid_horizon": int(item["valid_horizon"].item()),
                    "actual_frames": int(item["actual_frames"].item()),
                }
            )
            if args.max_items is not None and len(bank) >= args.max_items:
                break
        if args.max_items is not None and len(bank) >= args.max_items:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path)

    if args.index_json is not None:
        index_path = Path(args.index_json)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_records, f, indent=2)

    print(f"Saved {len(bank)} condition items to {output_path}")


if __name__ == "__main__":
    main()
