"""Analyze task distribution and propose balanced train/test split.

Usage:
  python scripts/analyze_task_distribution.py \
      --scene-info /data1/zmh/stage1_main_spatial_task_prior_franka/scene_info.json \
      --csv /data1/zmh/robotwin_sft_vanilla/metadata.csv \
      --train-core 0-599 \
      --train-ratio 0.8
"""
import argparse
import csv
import json
import re
import random
from collections import defaultdict
from pathlib import Path


def episode_num(ep_id: str) -> int:
    m = re.search(r'(\d+)', ep_id)
    return int(m.group(1)) if m else -1


def parse_range(s: str) -> set:
    result = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze task distribution")
    parser.add_argument("--scene-info", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--train-core", default=None,
                        help="Episode range forced into train, e.g. '0-599'")
    parser.add_argument("--test-core", default=None,
                        help="Episode range forced into test, e.g. '600-1199'")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Target train ratio (default: 0.8)")
    parser.add_argument("--min-episodes", type=int, default=0,
                        help="Only include tasks with >= N episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Save split to JSON file, e.g. 'train_test_split.json'")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load scene_info
    with open(args.scene_info, "r") as f:
        scene_info = json.load(f)

    # Force-core: episodes that MUST go to specific split
    core_train_ids = parse_range(args.train_core) if args.train_core else set()
    core_test_ids = parse_range(args.test_core) if args.test_core else set()
    # Conflict: train-core wins
    core_test_ids -= core_train_ids

    # Build per-task episode lists, sorted by episode number
    task_eps = defaultdict(list)
    for ep_id, ep_data in scene_info.items():
        task = ep_data.get("source_task", "unknown")
        num = episode_num(ep_id)
        task_eps[task].append((num, ep_id))

    # Sort each task's episodes
    for task in task_eps:
        task_eps[task].sort()

    # Filter by min episodes
    if args.min_episodes > 0:
        task_eps = {t: eps for t, eps in task_eps.items()
                    if len(eps) >= args.min_episodes}

    total = sum(len(v) for v in task_eps.values())
    n_tasks = len(task_eps)

    # ====== BALANCED SPLIT ======
    # Strategy:
    #   1. For each task, core-range episodes → train
    #   2. Remaining episodes split to achieve ~train_ratio per task
    #   3. This gives balanced task distribution in train and test

    train_set = set()
    test_set = set()

    core_total = len(core_train_ids) + len(core_test_ids)

    for task, eps in task_eps.items():
        core_tr = [(n, eid) for n, eid in eps if n in core_train_ids]
        core_te = [(n, eid) for n, eid in eps if n in core_test_ids]
        rest_eps = [(n, eid) for n, eid in eps
                    if n not in core_train_ids and n not in core_test_ids]

        # Core assignments
        for _, eid in core_tr:
            train_set.add(eid)
        for _, eid in core_te:
            test_set.add(eid)

        # Rest: split to hit train_ratio per task
        n_core_tr = len(core_tr)
        n_core_te = len(core_te)
        target_train = int(len(eps) * args.train_ratio)
        n_rest_train = max(0, target_train - n_core_tr)
        n_rest_train = max(0, min(n_rest_train, len(rest_eps)))

        random.shuffle(rest_eps)
        for i, (_, eid) in enumerate(rest_eps):
            if i < n_rest_train:
                train_set.add(eid)
            else:
                test_set.add(eid)

    train_total = len(train_set)
    test_total = len(test_set)
    actual_ratio = train_total / (train_total + test_total) if (train_total + test_total) > 0 else 0

    print(f"Split: {train_total} train / {test_total} test ({actual_ratio:.1%})")
    if core_train_ids:
        train_core_matched = len(core_train_ids & {episode_num(eid) for eid in train_set})
        print(f"Train-core ({len(core_train_ids)}): {train_core_matched} in train")
    if core_test_ids:
        test_core_matched = len(core_test_ids & {episode_num(eid) for eid in test_set})
        print(f"Test-core  ({len(core_test_ids)}): {test_core_matched} in test")
    print()

    # Per-task table
    print("=" * 85)
    header = f"{'Task':<38s} {'Total':>6s} {'Train':>6s} {'Test':>6s} {'T%':>6s}"
    print(header)
    print("-" * 85)

    for task in sorted(task_eps):
        eps = task_eps[task]
        n_total = len(eps)
        n_train = sum(1 for _, eid in eps if eid in train_set)
        n_test = sum(1 for _, eid in eps if eid in test_set)
        t_pct = n_train / n_total * 100 if n_total > 0 else 0
        print(f"{task:<38s} {n_total:>6d} {n_train:>6d} {n_test:>6d} {t_pct:>5.1f}%")

    print("-" * 85)
    print(f"{'TOTAL':<38s} {total:>6d} {train_total:>6d} {test_total:>6d} "
          f"{actual_ratio:>5.1%}")

    # Train-only and test-only tasks
    train_only = []
    test_only = []
    both = []
    for task in sorted(task_eps):
        n_train = sum(1 for _, eid in task_eps[task] if eid in train_set)
        n_test = sum(1 for _, eid in task_eps[task] if eid in test_set)
        if n_train > 0 and n_test == 0:
            train_only.append(task)
        elif n_test > 0 and n_train == 0:
            test_only.append(task)
        else:
            both.append(task)

    print(f"\n=== Task balance ===")
    print(f"Tasks in BOTH train & test: {len(both)}/{n_tasks}  <- balanced")
    print(f"Tasks ONLY in train:       {len(train_only)}/{n_tasks}")
    print(f"Tasks ONLY in test:        {len(test_only)}/{n_tasks}")
    if train_only:
        print(f"  Train-only: {', '.join(train_only)}")
    if test_only:
        print(f"  Test-only: {', '.join(test_only)}")

    # Category summary
    print(f"\n=== Task category balance ===")
    cats = defaultdict(lambda: {"train": 0, "test": 0, "tasks": []})
    for task in sorted(task_eps):
        cat = task.split("_")[0]
        cats[cat]["tasks"].append(task)
        for _, eid in task_eps[task]:
            if eid in train_set:
                cats[cat]["train"] += 1
            else:
                cats[cat]["test"] += 1
    for cat in sorted(cats):
        c = cats[cat]
        total_cat = c["train"] + c["test"]
        r = c["train"] / total_cat * 100 if total_cat > 0 else 0
        print(f"  {cat:<15s}: {c['train']:>5d} train / {c['test']:>5d} test  "
              f"({r:.0f}% train)  tasks: {c['tasks']}")

    # Print train/test episode ID lists for use in data preparation
    print(f"\n=== Episode ID lists ===")
    print(f"Train episodes: {sorted(episode_num(e) for e in train_set)}")
    print(f"Test episodes:  {sorted(episode_num(e) for e in test_set)}")

    # CSV cross-reference
    csv_train_nums = set()
    csv_test_nums = set()
    if args.csv:
        csv_path = Path(args.csv)
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))
            csv_nums = set()
            for r in records:
                stem = Path(r["video_path"]).stem.rsplit("_w", 1)[0]
                m = re.search(r'(\d+)', stem)
                if m:
                    csv_nums.add(int(m.group(1)))
            csv_train_nums = csv_nums & {episode_num(e) for e in train_set}
            csv_test_nums = csv_nums & {episode_num(e) for e in test_set}
            print(f"\nCSV: {len(records)} records")
            print(f"  In train set: {len(csv_train_nums)}")
            print(f"  In test set:  {len(csv_test_nums)}")

    # Save split file
    if args.output:
        output = {
            "train_ratio": args.train_ratio,
            "train_core": args.train_core,
            "test_core": args.test_core,
            "train_total": train_total,
            "test_total": test_total,
            "train_episodes": sorted(episode_num(e) for e in train_set),
            "test_episodes": sorted(episode_num(e) for e in test_set),
            "tasks": {}
        }
        for task in sorted(task_eps):
            eps = task_eps[task]
            output["tasks"][task] = {
                "total": len(eps),
                "train": sorted(episode_num(eid) for _, eid in eps if eid in train_set),
                "test": sorted(episode_num(eid) for _, eid in eps if eid in test_set),
            }
        if csv_train_nums:
            output["csv_train_episodes"] = sorted(csv_train_nums)
            output["csv_test_episodes"] = sorted(csv_test_nums)

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSplit saved to: {args.output}")


if __name__ == "__main__":
    main()
