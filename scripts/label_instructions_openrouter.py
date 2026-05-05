"""Label RoboTwin HDF5 episodes with language instructions via OpenRouter MLLM.

Samples 3-4 frames per episode (head_camera), sends to a multimodal model,
and saves the generated instruction alongside the episode.

Usage:
  export OPENROUTER_API_KEY="sk-..."

  python scripts/label_instructions_openrouter.py \
    --data-dir /data1/zmh/stage1_main_spatial_task_prior_franka/data \
    --meta-json /data1/zmh/stage1_main_spatial_task_prior_franka/scene_info.json \
    --output-dir /data1/zmh/stage1_main_spatial_task_prior_franka/instructions

  # Resume from where left off (skips episodes already processed)
  python scripts/label_instructions_openrouter.py ... --resume

  # Different model / limit for testing
  python scripts/label_instructions_openrouter.py ... --model openai/gpt-4o-mini --max-episodes 10
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image
from openai import OpenAI

SYSTEM_PROMPT = """You are a robot instruction writer. Given frames from a dual-arm Franka
manipulation video and an expected task description, output ONE concise instruction sentence.

VERIFY the expected description against the actual video frames:
- Correct hand assignments if the video shows a different arm doing the action.
- Use the EXACT objects and hand you see in the frames.
- Keep the same action verb and task structure as the expected description unless the video contradicts it.

Output ONLY the instruction sentence, no quotes, no prefixes."""

USER_PROMPT_TEMPLATE = """Expected task: {source_task}
Description: {task_description}
Meta: {task_info}

{n_frames} video frames. Output ONE verified instruction sentence based on what you see."""


def decode_rgb_frame(encoded) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array."""
    if isinstance(encoded, np.ndarray) and encoded.dtype == np.uint8 and encoded.ndim == 3:
        return encoded
    img = Image.open(BytesIO(bytes(encoded)))
    return np.array(img.convert("RGB"))


def frame_to_b64(frame: np.ndarray) -> str:
    """Encode RGB numpy array to base64 data URI."""
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def sample_frames(
    hdf5_path: Path,
    camera: str = "head_camera",
    n_frames: int = 4,
) -> list[str]:
    """Sample evenly-spaced frames from an HDF5 episode, return as base64 strings."""
    with h5py.File(hdf5_path, "r") as f:
        ds = f[f"/observation/{camera}/rgb"]
        T = len(ds)
        if T <= n_frames:
            indices = list(range(T))
        else:
            indices = np.linspace(0, T - 1, n_frames, dtype=int).tolist()

        frames_b64 = []
        for idx in indices:
            frame = decode_rgb_frame(ds[idx])
            frames_b64.append(frame_to_b64(frame))
        return frames_b64


def build_messages(frames_b64: list[str], source_task: str, task_info: dict,
                    task_description: str = "") -> list[dict]:
    """Build the message payload for the multimodal API, including task priors."""
    task_info_str = json.dumps(task_info) if task_info else "(none)"
    content = [{"type": "text", "text": USER_PROMPT_TEMPLATE.format(
        n_frames=len(frames_b64), source_task=source_task,
        task_description=task_description or "(see meta)",
        task_info=task_info_str,
    )}]
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def label_episode(
    client: OpenAI,
    model: str,
    hdf5_path: Path,
    source_task: str = "unknown",
    task_info: dict | None = None,
    task_description: str = "",
    max_retries: int = 3,
) -> Optional[str]:
    """Generate a language instruction for one episode."""
    try:
        frames_b64 = sample_frames(hdf5_path)
    except Exception as e:
        print(f"  [WARN] Failed to sample frames: {e}", file=sys.stderr)
        return None

    messages = build_messages(frames_b64, source_task, task_info or {}, task_description)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=128,
                temperature=0.3,
            )
            text = resp.choices[0].message.content.strip()
            # Remove quotes that some models add
            text = text.strip('"').strip("'")
            return text
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                wait = min(2 ** attempt * 5, 60)
                print(f"  Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"  [WARN] API error (attempt {attempt+1}): {e}", file=sys.stderr)
                time.sleep(2)
    return None


# Official RoboTwin task descriptions (from task_config.py), used as MLLM prior.
OFFICIAL_TASK_DESCRIPTIONS: dict[str, str] = {
    "beat_block_hammer": "Pick up the hammer and use it to beat the block on the table. Use right arm if block x>0, else left arm.",
    "pick_dual_bottles": "Use both arms to simultaneously pick up two bottles and move them to the front target locations.",
    "pick_diverse_bottles": "Use both arms to simultaneously pick up two diverse bottles and move them to the front target locations.",
    "handover_block": "Use the left arm to grab the block and handover to the right arm, then use right arm to place it on the target block.",
    "stack_blocks_two": "Use one arm to pick up the first block and move it to target, then pick up the second block and stack it on the first.",
    "stack_blocks_three": "Move block1 to target, then stack block2 on block1, then stack block3 on block2.",
    "place_container_plate": "Pick up the container and place it on the plate.",
    "place_empty_cup": "Pick up the empty cup and place it on the coaster.",
    "place_shoe": "Pick up the shoe and place it on the target block, with the shoe head pointing left.",
    "place_dual_shoes": "Use both arms to pick up two shoes and place them onto the shoebox, shoe tips pointing left.",
    "adjust_bottle": "Pick up the bottle headup with the correct arm and place it at the target pose.",
    "blocks_ranking_rgb": "Place the red, green, and blue blocks in order from left to right in a row.",
    "blocks_ranking_size": "Arrange three blocks from largest to smallest, left to right.",
    "click_bell": "Click the bell's top center on the table.",
    "grab_roller": "Use both arms to grab the roller on the table and lift it upward.",
    "lift_pot": "Use both arms to lift the pot from the table.",
    "move_can_pot": "Pick up the can and move it beside the pot on the table.",
    "move_playingcard_away": "Pick up the playing card and move it horizontally away.",
    "move_stapler_pad": "Move the stapler to a colored mat with alignment.",
    "click_alarmclock": "Click the alarm clock's top center button on the table.",
    "dump_bin_bigbin": "Grab the small bin and pour the balls into the big bin.",
    "handover_mic": "Grasp the microphone with one arm and handover it to the other arm.",
    "hanging_mug": "Use left arm to pick up the mug and adjust its pose, then use right arm to pick it up and hang it onto the rack.",
    "move_pillbottle_pad": "Pick up the pillbottle and place it onto the pad.",
    "place_a2b_left": "Place the object on the left side of the target object.",
    "place_a2b_right": "Place the object on the right side of the target object.",
    "place_bread_basket": "Grab the bread and put it in the basket. Use both arms if two breads.",
    "place_bread_skillet": "Grab the bread and put it into the skillet.",
    "place_can_basket": "Pick up the can and put it into the basket, then use the other arm to lift the basket.",
    "place_cans_plasticbox": "Use both arms to pick and place cans into the plastic box.",
    "place_fan": "Grab the fan and place it on a colored pad with alignment.",
    "place_burger_fries": "Use both arms to pick the hamburg and frenchfries and put them onto the tray.",
    "place_mouse_pad": "Grab the mouse and place it on a colored pad.",
    "place_object_basket": "Grab the object and put it in the basket, then use the other arm to lift the basket away.",
    "place_object_scale": "Grab the object and put it on the scale.",
    "place_object_stand": "Place the object on the display stand.",
    "place_phone_stand": "Pick up the phone and put it on the phone stand.",
    "press_stapler": "Press the stapler.",
    "rotate_qrcode": "Pick up the QR code board and rotate it so the QR code faces forward.",
    "scan_object": "Use one arm to hold the scanner and the other arm to hold the object, and scan.",
    "stack_bowls_three": "Stack the three bowls on top of each other.",
    "stack_bowls_two": "Stack the two bowls on top of each other.",
    "stamp_seal": "Pick up the stamp and place it on the target block.",
    "shake_bottle_horizontally": "Shake the bottle horizontally with the proper arm.",
    "shake_bottle": "Shake the bottle up and down with the proper arm.",
    "turn_switch": "Click the switch with one arm.",
    "open_laptop": "Open the laptop with one arm.",
    "open_microwave": "Pull the handle to open the microwave with one arm.",
    "put_object_cabinet": "Use one arm to open the cabinet, and use the other arm to pick the object and put it into the cabinet.",
}


def main():
    parser = argparse.ArgumentParser(description="Label RoboTwin episodes with instructions via OpenRouter")
    parser.add_argument("--data-dir", required=True, help="Directory containing episode*.hdf5")
    parser.add_argument("--meta-json", default=None, help="Optional scene_info.json for context")
    parser.add_argument("--output-dir", required=True, help="Directory to save per-episode .txt instructions")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="OpenRouter model ID (default: openai/gpt-4o-mini)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit episodes (for testing)")
    parser.add_argument("--resume", action="store_true", help="Skip episodes that already have instruction files")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls to avoid rate limits")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENROUTER_API_KEY env)")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="API base URL")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(data_dir.glob("episode*.hdf5"))
    if not hdf5_files:
        print(f"ERROR: No episode*.hdf5 files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_episodes:
        hdf5_files = hdf5_files[:args.max_episodes]

    # Load metadata for task priors
    meta = {}
    if args.meta_json:
        with open(args.meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Loaded metadata: {len(meta)} entries")

    # Collect results for JSON output
    instructions = {}
    json_path = output_dir / "instructions.json"

    if args.resume and json_path.exists():
        with open(json_path, "r") as f:
            instructions = json.load(f)
        print(f"Resuming: {len(instructions)} already labeled")

    n_done = 0
    n_failed = 0
    n_skipped = 0

    for ep_path in hdf5_files:
        ep_id = ep_path.stem  # e.g., "episode_0"

        if args.resume and ep_id in instructions:
            n_skipped += 1
            continue

        # Look up task metadata for this episode
        # HDF5 stem: "episode0", meta key: "episode_0" — normalize to meta key format
        m = re.search(r'(\d+)', ep_id)
        ep_num = int(m.group(1)) if m else 0
        ep_meta = meta.get(f"episode_{ep_num}") or meta.get(str(ep_num)) or {}
        source_task = ep_meta.get("source_task", "unknown")
        task_info = ep_meta.get("task_info", {})

        print(f"[{n_done + n_failed + n_skipped + 1}/{len(hdf5_files)}] {ep_id} ({source_task}) ...", end=" ", flush=True)

        task_desc = OFFICIAL_TASK_DESCRIPTIONS.get(source_task, "")
        instruction = label_episode(client, args.model, ep_path, source_task, task_info, task_desc)
        if instruction:
            instructions[ep_id] = instruction
            # Also save individual .txt
            txt_path = output_dir / f"{ep_id}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(instruction)
            print(f"OK: {instruction}")
            n_done += 1
        else:
            print("FAILED")
            n_failed += 1

        # Save progress periodically
        if n_done % 20 == 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(instructions, f, indent=2, ensure_ascii=False)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Final save
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {n_done} labeled, {n_failed} failed, {n_skipped} skipped")
    print(f"Instructions saved to {json_path} and {output_dir}/")


if __name__ == "__main__":
    main()
