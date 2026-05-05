"""Prepare Robotwin SFT dataset for Wan2.1 training.

Two modes:
  --mode vanilla    Full episodes, one record per episode (EVA baseline)
  --mode arm-entry  Arm-entry windows only (our entry-focused SFT)

Arm-entry detection: uses qpos data (gold truth). A window is "arm-entry" if
the shoulder/elbow joints change significantly from first to last frame,
indicating the arm transitions from retracted (out of view) to extended (in view).

Usage:
  # Vanilla SFT dataset (with composite 3-view stitching)
  python scripts/prepare_robotwin_sft_data.py \
    --data-dir /path/to/hdf5_episodes --meta-json /path/to/meta.json \
    --output-dir /path/to/robotwin_sft_vanilla --mode vanilla --composite

  # Arm-entry SFT dataset
  python scripts/prepare_robotwin_sft_data.py \
    --data-dir /path/to/hdf5_episodes --meta-json /path/to/meta.json \
    --output-dir /path/to/robotwin_sft_entry --mode arm-entry --composite
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Task Templates ──────────────────────────────────────────────────────────

# Templates derived directly from official RoboTwin task_info.py task_description.
# {A}/{B}/{C} → objects in actor_list order (mapped in scene_info.json)
# {a}/{b}/{c} → hand assignments (left/right, from qpose_tag or task convention)
# Placeholder POSITION in the sentence follows the official description's semantic role.
# MLLM instructions (via --instructions-dir) override these.
TASK_TEMPLATES: Dict[str, str] = {
    # ── Placement ──────────────────────────────────────────────────────
    "adjust_bottle": "Pick up the {A} headup with the {a} hand and place it at the target pose.",
    "place_a2b_left": "Use the {a} hand to place the {A} on the left of the {B}.",
    "place_a2b_right": "Use the {a} hand to place the {A} on the right of the {B}.",
    "place_bread_basket": "Use the {a} hand to grab the {B} and put it in the {A}.",
    "place_bread_skillet": "Use the {a} hand to grab the {B} and put it into the {A}.",
    "place_burger_fries": "Use both arms to pick the {A} and {C} and put them onto the {B}.",
    "place_can_basket": "Use the {a} hand to pick up the {A} and place it into the {B}, then use the other arm to lift the {B}.",
    "place_cans_plasticbox": "Use both arms to pick and place the {A} and {C} into the {B}.",
    "place_container_plate": "Use the {a} hand to pick up the {B} and place it in the {A}.",
    "place_dual_shoes": "Use both arms to pick up the {A} and place them onto the {B}, shoe tips pointing left.",
    "place_empty_cup": "Use the {a} hand to pick up the {A} and place it on the {B}.",
    "place_fan": "Grab the {A} with the {a} hand and place it on the {B} pad.",
    "place_mouse_pad": "Grab the {A} with the {a} hand and place it on the {B} pad.",
    "place_object_basket": "Use the {a} hand to grab the {A} and put it in the {B}, then use the {b} hand to lift the {B} away.",
    "place_object_scale": "Use the {a} hand to grab the {B} and put it on the {A}.",
    "place_object_stand": "Use the {a} hand to place the {A} on the {B}.",
    "place_phone_stand": "Pick up the {A} with the {a} hand and put it on the {B}.",
    "place_shoe": "Pick up the {A} with the {a} hand and place it on the target block, shoe head pointing left.",
    "put_object_cabinet": "Use the {a} hand to open the {B}, and use the {b} hand to pick the {A} and put it into the {B}.",

    # ── Pick / Move ────────────────────────────────────────────────────
    "move_can_pot": "Use the {a} hand to pick up the {B} and move it beside the {A}.",
    "move_pillbottle_pad": "Pick the {A} with the {a} hand and place it onto the pad.",
    "move_playingcard_away": "Use the {a} hand to pick up the {A} and move it horizontally away.",
    "move_stapler_pad": "Use the {a} hand to move the {A} to the {B} pad.",
    "pick_diverse_bottles": "Use both arms to simultaneously pick up the {A} and {B} and move them to the front target locations.",
    "pick_dual_bottles": "Use both arms to simultaneously pick up the {A} and {B} and move them to the front target locations.",
    "grab_roller": "Use both arms to grab the {A} on the table and lift it upward.",
    "lift_pot": "Use both arms to lift the {A}.",

    # ── Stacking ───────────────────────────────────────────────────────
    "stack_blocks_two": "Move the {A} to the target, then stack the {B} on top of the {A}.",
    "stack_blocks_three": "Move the {A} to the target, stack the {B} on the {A}, stack the {C} on the {B}.",
    "stack_bowls_two": "Stack the {B} on top of the {A}.",
    "stack_bowls_three": "Stack the three bowls on top of each other.",

    # ── Ranking ────────────────────────────────────────────────────────
    "blocks_ranking_rgb": "Place the {A}, {B}, and {C} in order from left to right.",
    "blocks_ranking_size": "Arrange the {A}, {B}, and {C} from largest to smallest, left to right.",

    # ── Tool use ───────────────────────────────────────────────────────
    "beat_block_hammer": "Pick up the {A} and use it to beat the block on the table.",
    "click_alarmclock": "Click the {A} with the {a} hand.",
    "click_bell": "Click the {A} with the {a} hand.",
    "open_laptop": "Open the {A} with the {a} hand.",
    "open_microwave": "Pull the handle of the {A} with the {a} hand to open it.",
    "press_stapler": "Use the {a} hand to press the {A}.",
    "stamp_seal": "Pick the {A} with the {a} hand and place it on the target block.",
    "turn_switch": "Click the {A} with the {a} hand.",

    # ── Handover ───────────────────────────────────────────────────────
    "handover_block": "Use the left arm to grasp the block and handover to the right arm, then place it on the target block.",
    "handover_mic": "Use the {a} hand to grasp the {A} and handover it to the {b} hand.",

    # ── Pour / Shake ───────────────────────────────────────────────────
    "dump_bin_bigbin": "Grab the {A} and pour the balls into the big bin.",
    "shake_bottle": "Use the {a} hand to shake the {A}.",
    "shake_bottle_horizontally": "Use the {a} hand to shake the {A} horizontally.",

    # ── Special ────────────────────────────────────────────────────────
    "hanging_mug": "Use the left arm to pick up the {A} and adjust its pose, then use the right arm to pick it up and hang it onto the {B}.",
    "rotate_qrcode": "Use the {a} hand to pick up the {A} and rotate it so the QR code faces forward.",
    "scan_object": "Use the {b} hand to pick the {B} and use the {a} hand to pick the {A}, then scan the object.",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def object_name_from_path(obj_path: str) -> str:
    if not obj_path:
        return "object"
    part = obj_path.split("/")[0]
    name = re.sub(r'^\d+_', '', part)
    return name.replace("_", " ")


def build_caption(source_task: str, task_info: Dict[str, str]) -> str:
    template = TASK_TEMPLATES.get(source_task)
    if template is None:
        return f"A Franka robot arm performing {source_task.replace('_', ' ')}."
    subs = {}
    for ph, raw in task_info.items():
        if ph in ("{a}", "{b}", "{c}"):
            subs[ph] = raw
        elif ph in ("{A}", "{B}", "{C}"):
            subs[ph] = object_name_from_path(raw)
    caption = template
    for ph, val in subs.items():
        caption = caption.replace(ph, val)
    # Clean up any unreplaced placeholders
    caption = re.sub(r'\{[a-cA-C]\}', '', caption).strip()
    caption = re.sub(r'\s{2,}', ' ', caption)  # collapse double spaces
    caption = re.sub(r'\s,', ',', caption)      # fix ", " artifacts
    return caption


def decode_rgb_frame(encoded) -> np.ndarray:
    if isinstance(encoded, np.ndarray) and encoded.dtype == np.uint8 and encoded.ndim == 3:
        image = encoded
    else:
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RGB frame.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_mp4_segment(
    hdf5_path: Path,
    output_mp4_path: Path,
    start_frame: int,
    end_frame: int,
    camera_name: str = "head_camera",
    resize: Optional[tuple[int, int]] = None,
    fps: int = 16,
    composite: bool = False,
) -> int:
    """Extract a frame range from HDF5 and write as MP4.

    If composite=True, stitch head_camera + left_camera + right_camera into
    a single 640×720 frame matching the IDM's Aloha-style input format:
      top half:    head 640×480  (resized from 320×240)
      bottom half: left 320×240 + right 320×240 side by side → 640×240
    """
    try:
        with h5py.File(hdf5_path, "r") as root:
            head_ds = root[f"/observation/{camera_name}/rgb"]
            n_frames = min(end_frame, len(head_ds)) - start_frame
            if n_frames <= 0:
                return 0

            if composite:
                left_ds = root["/observation/left_camera/rgb"]
                right_ds = root["/observation/right_camera/rgb"]
                out_w, out_h = 640, 720
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (out_w, out_h))
                for t in range(start_frame, min(end_frame, len(head_ds))):
                    head = cv2.resize(decode_rgb_frame(head_ds[t]), (640, 480))
                    left = decode_rgb_frame(left_ds[t])   # 320×240
                    right = decode_rgb_frame(right_ds[t])  # 320×240
                    bottom = np.concatenate([left, right], axis=1)  # 640×240
                    frame = np.concatenate([head, bottom], axis=0)  # 640×720
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                return min(end_frame, len(head_ds)) - start_frame
            else:
                first = decode_rgb_frame(head_ds[start_frame])
                h, w = first.shape[:2]
                if resize is not None:
                    w, h = resize

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (w, h))
                for t in range(start_frame, min(end_frame, len(head_ds))):
                    frame = decode_rgb_frame(head_ds[t])
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if resize is not None:
                        frame_bgr = cv2.resize(frame_bgr, resize, interpolation=cv2.INTER_LINEAR)
                    writer.write(frame_bgr)
                writer.release()
                return min(end_frame, len(head_ds)) - start_frame
    except Exception as e:
        print(f"  [WARN] {hdf5_path.name}[{start_frame}:{end_frame}]: {e}", file=sys.stderr)
        return 0


# ── Arm-entry detection ─────────────────────────────────────────────────────
# Left arm: joints [0-6], Right arm: joints [8-14]
# Shoulder joints (index 0-2 for left, 8-10 for right) control arm position.
# Large deltas → arm entering or leaving the camera view.

LEFT_ARM = slice(0, 7)
RIGHT_ARM = slice(8, 15)
SHOULDER_LEFT = slice(0, 3)
SHOULDER_RIGHT = slice(8, 11)


def is_arm_entry_window(
    qpos: np.ndarray,
    start: int,
    end: int,
    arm_delta_thresh: float = 0.3,
) -> bool:
    """Check if a window is an arm-entry window using qpos deltas.

    Returns True if either left or right arm shoulder joints have significant
    displacement from start to end of window, indicating arm movement into view.
    """
    if end > qpos.shape[0]:
        return False
    delta = np.abs(qpos[end - 1] - qpos[start])
    left_shoulder_delta = delta[SHOULDER_LEFT].mean()
    right_shoulder_delta = delta[SHOULDER_RIGHT].mean()
    return left_shoulder_delta > arm_delta_thresh or right_shoulder_delta > arm_delta_thresh


# ── Episode processing ──────────────────────────────────────────────────────

def process_vanilla(
    data_dir: Path,
    meta: Dict[str, Any],
    output_dir: Path,
    camera_name: str,
    resize: Optional[tuple[int, int]],
    fps: int,
    val_split: float,
    seed: int,
    from_mp4: bool,
    composite: bool = False,
    instructions: Optional[Dict[str, str]] = None,
) -> list[dict]:
    """Process full episodes — one record per episode."""
    if from_mp4:
        mp4_files = sorted(data_dir.glob("episode*.mp4"))
        mp4_files = [p for p in mp4_files if "depth" not in p.name]
        if not mp4_files:
            raise FileNotFoundError(f"No episode*.mp4 files in {data_dir}")

        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for mp4_path in tqdm(mp4_files, desc="Vanilla (MP4)"):
            ep_id = _extract_ep_id(mp4_path.stem)
            ep_key = f"episode_{ep_id}"
            ep_meta = meta.get(ep_key) or meta.get(str(ep_id)) or {}
            caption = _get_caption(instructions, ep_key, ep_meta)
            cap = cv2.VideoCapture(str(mp4_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            records.append({
                "video_path": str(mp4_path.relative_to(output_dir)),
                "caption": caption, "height": h, "width": w,
                "n_frames": n_frames, "fps": fps,
                "source_task": ep_meta.get("source_task", "unknown"),
                "episode_id": str(ep_id),
            })
    else:
        hdf5_files = sorted(data_dir.glob("episode*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files in {data_dir}")

        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for ep_path in tqdm(hdf5_files, desc="Vanilla (HDF5)"):
            ep_id = _extract_ep_id(ep_path.stem)
            ep_key = f"episode_{ep_id}"
            ep_meta = meta.get(ep_key) or meta.get(str(ep_id)) or {}
            caption = _get_caption(instructions, ep_key, ep_meta)
            mp4_name = f"episode_{ep_id:06d}.mp4"
            mp4_path = video_dir / mp4_name
            if not mp4_path.exists():
                n_frames = write_mp4_segment(ep_path, mp4_path, 0, 10**9, camera_name, resize, fps, composite=composite)
                if n_frames == 0:
                    continue
            n_frames = _get_mp4_frames(mp4_path)
            h = 720 if composite else (resize[1] if resize else 480)
            w = 640 if composite else (resize[0] if resize else 640)
            records.append({
                "video_path": str(mp4_path.relative_to(output_dir)),
                "caption": caption, "height": h, "width": w,
                "n_frames": n_frames, "fps": fps,
                "source_task": ep_meta.get("source_task", "unknown"),
                "episode_id": str(ep_id),
            })

    return _assign_split(records, val_split, seed)


def process_arm_entry(
    data_dir: Path,
    meta: Dict[str, Any],
    output_dir: Path,
    camera_name: str,
    resize: Optional[tuple[int, int]],
    fps: int,
    n_frames_target: int,
    val_split: float,
    seed: int,
    arm_delta_thresh: float,
    window_stride: int,
    max_windows_per_ep: int,
    composite: bool = False,
    instructions: Optional[Dict[str, str]] = None,
) -> list[dict]:
    """Process episodes — extract arm-entry windows only.

    Slides a window of n_frames_target (in source frames at 30fps) across each
    episode. For each window, checks if arm shoulder joints change significantly
    (indicating arm entry). Keeps qualifying windows.
    """
    hdf5_files = sorted(data_dir.glob("episode*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {data_dir}")

    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    records = []
    window_src_frames = n_frames_target * 30 // fps  # e.g. 17 * 30/16 ≈ 32 source frames

    for ep_path in tqdm(hdf5_files, desc="Arm-entry"):
        ep_id = _extract_ep_id(ep_path.stem)
        ep_key = f"episode_{ep_id}"
        ep_meta = meta.get(ep_key) or meta.get(str(ep_id)) or {}
        caption = _get_caption(instructions, ep_key, ep_meta)

        # Read qpos for arm-entry detection
        try:
            with h5py.File(ep_path, "r") as root:
                for qpos_key in ["/joint_action/vector", "/qpos", "/robot_state/qpos"]:
                    if qpos_key in root:
                        qpos = np.asarray(root[qpos_key], dtype=np.float32)
                        break
                else:
                    continue  # skip if no qpos
                ep_len = qpos.shape[0]
        except Exception:
            continue

        n_windows = 0
        for start in range(0, max(1, ep_len - window_src_frames), window_stride):
            end = start + window_src_frames
            if end > ep_len:
                break
            if not is_arm_entry_window(qpos, start, end, arm_delta_thresh):
                continue

            mp4_name = f"episode_{ep_id:06d}_w{start:04d}.mp4"
            mp4_path = video_dir / mp4_name
            if not mp4_path.exists():
                n_out = write_mp4_segment(ep_path, mp4_path, start, end, camera_name, resize, fps, composite=composite)
                if n_out == 0:
                    continue

            n_frames = _get_mp4_frames(mp4_path)
            h = 720 if composite else (resize[1] if resize else 480)
            w = 640 if composite else (resize[0] if resize else 640)
            records.append({
                "video_path": str(mp4_path.relative_to(output_dir)),
                "caption": caption, "height": h, "width": w,
                "n_frames": n_frames, "fps": fps,
                "source_task": ep_meta.get("source_task", "unknown"),
                "episode_id": str(ep_id),
            })
            n_windows += 1
            if max_windows_per_ep and n_windows >= max_windows_per_ep:
                break

    return _assign_split(records, val_split, seed)


# ── Utilities ───────────────────────────────────────────────────────────────

def _extract_ep_id(stem: str) -> int:
    m = re.search(r'episode[_\s]*(\d+)', stem, re.IGNORECASE)
    return int(m.group(1)) if m else stem


def _get_mp4_frames(mp4_path: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _get_caption(
    instructions: Optional[Dict[str, str]],
    ep_key: str,
    ep_meta: Dict[str, Any],
) -> str:
    """Return MLLM instruction if available, else fall back to template caption."""
    if instructions and ep_key in instructions:
        return instructions[ep_key]
    return build_caption(ep_meta.get("source_task", "unknown"), ep_meta.get("task_info", {}))


def _assign_split(records: list[dict], val_split: float, seed: int) -> list[dict]:
    rng = np.random.RandomState(seed)
    # Split by unique episode IDs to avoid data leakage
    ep_ids = sorted(set(r["episode_id"] for r in records))
    rng.shuffle(ep_ids)
    n_val = max(1, int(len(ep_ids) * val_split))
    val_eps = set(ep_ids[:n_val])
    for r in records:
        r["split"] = "validation" if r["episode_id"] in val_eps else "training"
    return records


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Robotwin SFT dataset")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--meta-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["vanilla", "arm-entry"], default="vanilla")
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--n-frames", type=int, default=17,
                        help="Target frames per sample (for arm-entry window sizing)")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from-mp4", action="store_true")
    parser.add_argument("--composite", action="store_true",
                        help="Stitch head+left+right cameras into 640×480 frame (Aloha-style)")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--arm-delta-thresh", type=float, default=0.3,
                        help="Min shoulder joint delta (rad) for arm-entry detection")
    parser.add_argument("--window-stride", type=int, default=8,
                        help="Frame stride for sliding window in arm-entry mode")
    parser.add_argument("--max-windows-per-ep", type=int, default=10,
                        help="Max arm-entry windows per episode")
    parser.add_argument("--instructions-dir", default=None,
                        help="Directory with MLLM-generated instructions.json (overrides template captions)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"Loaded metadata: {len(meta)} entries")

    resize = (args.width, args.height)

    # Load MLLM instructions if provided
    instructions = None
    if args.instructions_dir:
        instr_json = Path(args.instructions_dir) / "instructions.json"
        if instr_json.exists():
            with open(instr_json, "r", encoding="utf-8") as f:
                instructions = json.load(f)
            print(f"Loaded {len(instructions)} MLLM instructions from {instr_json}")
        else:
            print(f"[WARN] instructions.json not found in {args.instructions_dir}")

    if args.mode == "vanilla":
        records = process_vanilla(
            data_dir, meta, output_dir,
            camera_name=args.camera_name,
            resize=resize, fps=args.fps,
            val_split=args.val_split, seed=args.seed,
            from_mp4=args.from_mp4,
            composite=args.composite,
            instructions=instructions,
        )
    else:  # arm-entry
        records = process_arm_entry(
            data_dir, meta, output_dir,
            camera_name=args.camera_name,
            resize=resize, fps=args.fps,
            n_frames_target=args.n_frames,
            val_split=args.val_split, seed=args.seed,
            arm_delta_thresh=args.arm_delta_thresh,
            window_stride=args.window_stride,
            max_windows_per_ep=args.max_windows_per_ep,
            composite=args.composite,
            instructions=instructions,
        )

    if args.max_episodes and args.mode == "vanilla":
        records = records[:args.max_episodes]

    # Write CSV
    csv_path = output_dir / "metadata.csv"
    fieldnames = ["video_path", "caption", "height", "width", "n_frames", "fps", "split"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    n_train = sum(1 for r in records if r["split"] == "training")
    n_val = sum(1 for r in records if r["split"] == "validation")

    print(f"\nDone! {len(records)} records → {csv_path}")
    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"  Unique episodes: {len(set(r['episode_id'] for r in records))}")
    print("\nSample captions:")
    for rec in records[:5]:
        print(f"  [{rec['episode_id']}] {rec['caption']}")


if __name__ == "__main__":
    main()
