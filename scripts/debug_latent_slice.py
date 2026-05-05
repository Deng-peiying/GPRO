"""Debug: find which videos cause _load_video_latent_slice IndexError.

Root cause hypothesis:
  lat_end (= _frame_to_latent_idx(last) + 1) can exceed full_latent.shape[1],
  but Python slicing silently truncates. Then actual_t (= lat_end - lat_start)
  OVERESTIMATES the real slice size, causing index_select to go OOB.

Usage:
  python scripts/debug_latent_slice.py \
    --data-dir /data1/zmh/robotwin_sft_vanilla \
    --csv metadata.csv \
    --n-frames 25
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


def frame_to_latent_idx(frame_idx: int) -> int:
    if frame_idx <= 0:
        return 0
    return 1 + (frame_idx - 1) // 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--csv", default="metadata.csv")
    parser.add_argument("--n-frames", type=int, default=25)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / args.csv

    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print(f"Loaded {len(records)} records")
    print(f"n_frames={args.n_frames}, fps={args.fps}")
    print()

    # We can't use decord without GPU; instead read n_frames from record metadata
    # and cross-reference with latent file shape.
    #
    # For speedup mode with n_video >= target_len:
    #   frame_indices = linspace(0, n_video-1, n_frames)
    #   last = n_video - 1  (exact: linspace endpoint equals stop)
    #   lat_end = frame_to_latent_idx(n_video-1) + 1
    #
    # Latent T_lat should match: frame_to_latent_idx(n_video-1) + 1
    #   = 2 + (n_video-2)//4  for n_video > 1

    mismatches = []
    oob_cases = []
    ok = 0
    missing_latent = 0

    for i, rec in enumerate(records):
        video_name = Path(rec["video_path"]).stem
        latent_path = data_dir / rec.get("video_latent_path", "")
        if not latent_path.exists():
            missing_latent += 1
            continue

        full_latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        T_lat = full_latent.shape[1]

        # Get video frame count from record
        n_video = int(rec.get("n_frames", 0))
        if n_video == 0:
            # Fallback: derive from latent shape
            # T_lat = 2 + (n_video-2)//4 => n_video = 4*(T_lat-2)+2 to 4*(T_lat-1)+1
            continue

        src_fps = float(rec.get("fps", args.fps))
        target_len = round(args.n_frames / args.fps * src_fps)

        if n_video < target_len:
            # pad_mode=discard should filter these, but check anyway
            continue

        # speedup branch: uniform over full video
        frame_indices = np.linspace(0, n_video - 1, args.n_frames)
        frame_indices = np.round(frame_indices).astype(int)

        first = int(frame_indices[0])
        last = int(frame_indices[-1])
        lat_start = frame_to_latent_idx(first)
        lat_end = frame_to_latent_idx(last) + 1

        # Simulate slice (Python slicing semantics)
        real_lat_end = min(lat_end, T_lat)
        real_actual_t = real_lat_end - lat_start
        computed_actual_t = lat_end - lat_start

        expected_t = 1 + (args.n_frames - 1) // 4

        # Check 1: lat_end exceeds T_lat
        if lat_end > T_lat:
            mismatches.append({
                "idx": i, "video": video_name,
                "n_video": n_video, "src_fps": src_fps,
                "target_len": target_len,
                "first_frame": first, "last_frame": last,
                "lat_start": lat_start, "lat_end": lat_end,
                "T_lat": T_lat, "delta": lat_end - T_lat,
                "computed_actual_t": computed_actual_t,
                "real_actual_t": real_actual_t,
                "expected_t": expected_t,
            })

        # Check 2: would subsampling indices go OOB?
        if computed_actual_t > expected_t:
            # This triggers the subsample path, but uses computed_actual_t
            indices = torch.linspace(0, computed_actual_t - 1, expected_t).round().long()
            indices = indices.clamp(0, computed_actual_t - 1)
            max_idx = indices.max().item()
            if max_idx >= real_actual_t:
                oob_cases.append({
                    "idx": i, "video": video_name,
                    "n_video": n_video,
                    "T_lat": T_lat,
                    "computed_actual_t": computed_actual_t,
                    "real_actual_t": real_actual_t,
                    "expected_t": expected_t,
                    "indices": indices.tolist(),
                    "max_idx": max_idx,
                })
            else:
                ok += 1
        else:
            ok += 1

    # Report
    print(f"OK: {ok}")
    print(f"Missing latent: {missing_latent}")
    print(f"Mismatches (lat_end > T_lat): {len(mismatches)}")
    print(f"OOB cases: {len(oob_cases)}")
    print()

    if mismatches:
        print("=" * 90)
        print("VIDEOS WHERE lat_end > T_lat (slice truncation):")
        print("=" * 90)
        for m in mismatches:
            print(f"  [{m['idx']}] {m['video']}")
            print(f"      n_video={m['n_video']} src_fps={m['src_fps']} target_len={m['target_len']}")
            print(f"      frames: first={m['first_frame']} last={m['last_frame']}")
            print(f"      lat_start={m['lat_start']} lat_end={m['lat_end']} T_lat={m['T_lat']}")
            print(f"      delta={m['delta']} computed_actual_t={m['computed_actual_t']} real_actual_t={m['real_actual_t']}")
            print(f"      expected_t={m['expected_t']}")
            print()

    if oob_cases:
        print("=" * 90)
        print("OOB CASES (indices would go out of bounds):")
        print("=" * 90)
        for c in oob_cases:
            print(f"  [{c['idx']}] {c['video']}")
            print(f"      n_video={c['n_video']} T_lat={c['T_lat']}")
            print(f"      computed_actual_t={c['computed_actual_t']} real_actual_t={c['real_actual_t']}")
            print(f"      expected_t={c['expected_t']} max_idx={c['max_idx']}")
            print(f"      indices={c['indices']}")
            print()

    if not mismatches and not oob_cases:
        print("No issues found with record metadata. The problem may be:")
        print("  1. decord len(vr) differs from record['n_frames'] at training time")
        print("  2. Some latent files are corrupted (shorter than expected)")
        print()
        print("Running deep scan: check EVERY latent file's T_lat directly...")
        print()

        # Deep scan: compute expected T_lat from each latent and check consistency
        deep_issues = []
        for i, rec in enumerate(records):
            latent_path_str = rec.get("video_latent_path", "")
            if not latent_path_str:
                continue
            latent_path = data_dir / latent_path_str
            if not latent_path.exists():
                continue
            full_latent = torch.load(latent_path, map_location="cpu", weights_only=True)
            T_lat = full_latent.shape[1]

            n_video = int(rec.get("n_frames", 0))
            if n_video == 0:
                continue

            expected_T_lat = frame_to_latent_idx(n_video - 1) + 1 if n_video > 0 else 1

            if T_lat != expected_T_lat:
                deep_issues.append({
                    "idx": i,
                    "video": Path(rec["video_path"]).stem,
                    "n_video": n_video,
                    "T_lat": T_lat,
                    "expected_T_lat": expected_T_lat,
                })

        if deep_issues:
            print(f"Found {len(deep_issues)} videos with T_lat mismatch:")
            for d in deep_issues[:20]:
                print(f"  [{d['idx']}] {d['video']}: n_video={d['n_video']} "
                      f"T_lat={d['T_lat']} expected={d['expected_T_lat']} "
                      f"delta={d['T_lat'] - d['expected_T_lat']}")
        else:
            print("All latent T_lat match expected values from record n_frames.")


if __name__ == "__main__":
    main()
