"""Rebuild 12x12 data as Layout -> Task.

Layouts are unique wall configurations. Tasks add start/goal pairs.

Split policy:
- L1-L4: split layouts; all tasks of a layout stay in one split.
- L0: one empty layout; split tasks by task hash.

This script reads the existing data/maps/*.jsonl as the source of wall
layouts, and writes data/layouts/ and data/tasks/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.maze import (  # noqa: E402
    INF, LEVEL_RULES, LEVELS, bfs_field, manhattan,
)

SPLITS = ("train", "val", "test")
SOURCE_SPLITS = ("train", "val", "test_id")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def layout_id(level: str, w: int, h: int, walls) -> str:
    payload = {
        "level": level,
        "w": int(w),
        "h": int(h),
        "walls": sorted([int(x), int(y)] for x, y in walls),
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def task_id(lid: str, start, goal) -> str:
    payload = {
        "layout_id": lid,
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]), int(goal[1])],
    }
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def load_layouts(source_dir: Path):
    layouts = {}
    for split in SOURCE_SPLITS:
        for row in load_jsonl(source_dir / "maps" / f"{split}.jsonl"):
            level = row["level"]
            lid = layout_id(level, row["w"], row["h"], row["walls"])
            if lid not in layouts:
                layouts[lid] = {
                    "layout_id": lid,
                    "level": level,
                    "width": int(row["w"]),
                    "height": int(row["h"]),
                    "max_steps": int(row.get("max_steps", 30)),
                    "walls": sorted([int(x), int(y)] for x, y in row["walls"]),
                    "source_id": row.get("id"),
                }
    return layouts


def select_layouts_with_tasks(layouts, *, train_n: int, val_n: int,
                              test_n: int, tasks_per_layout: int,
                              rng: random.Random):
    """Select only layouts that can produce enough valid tasks."""
    by_level = defaultdict(list)
    for layout in layouts.values():
        if layout["level"] != "L0":
            by_level[layout["level"]].append(layout)
    chosen = {split: [] for split in SPLITS}
    for level in LEVELS:
        if level == "L0":
            continue
        need = {"train": train_n, "val": val_n, "test": test_n}
        items = sorted(by_level[level], key=lambda x: x["layout_id"])
        cursor = 0
        level_chosen = {split: [] for split in SPLITS}
        for split in SPLITS:
            while len(level_chosen[split]) < need[split]:
                if cursor >= len(items):
                    raise RuntimeError(
                        f"{level} 可用 layout 不足，无法填满 {split} "
                        f"{need[split]} 个")
                layout = items[cursor]
                cursor += 1
                tasks = generate_tasks(layout, tasks_per_layout, rng)
                if len(tasks) < tasks_per_layout:
                    continue
                level_chosen[split].append((layout, tasks))
        for split in SPLITS:
            chosen[split].extend(level_chosen[split])
    return chosen


def generate_tasks(layout: dict, n: int, rng: random.Random):
    """Sample valid start/goal tasks for one layout."""
    level = layout["level"]
    w, h = layout["width"], layout["height"]
    walls = {tuple(p) for p in layout["walls"]}
    max_steps = layout["max_steps"]
    rules = LEVEL_RULES[level]
    free = [(x, y) for y in range(h) for x in range(w)
            if (x, y) not in walls]
    if len(free) < 2:
        return []

    out = []
    seen = set()
    attempts = 0
    max_attempts = max(2000, n * 500)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        start = rng.choice(free)
        goal = rng.choice(free)
        if start == goal:
            continue
        dist = bfs_field(w, h, walls, goal)
        d = dist[start[1]][start[0]]
        if d == INF or d < rules["min_dist"]:
            continue
        if d > max_steps - 2:
            continue
        detour = d - manhattan(start, goal)
        if detour < rules["min_detour"]:
            continue
        tid = task_id(layout["layout_id"], start, goal)
        if tid in seen:
            continue
        seen.add(tid)
        out.append({
            "task_id": tid,
            "layout_id": layout["layout_id"],
            "level": level,
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "shortest": int(d),
            "detour": int(detour),
            "hash": tid,
        })
    return out


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-layouts", type=int, default=120)
    ap.add_argument("--val-layouts", type=int, default=16)
    ap.add_argument("--test-layouts", type=int, default=16)
    ap.add_argument("--tasks-per-layout", type=int, default=5)
    ap.add_argument("--l0-train", type=int, default=600)
    ap.add_argument("--l0-val", type=int, default=80)
    ap.add_argument("--l0-test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    t0 = time.time()
    layouts = load_layouts(data)
    print(f"loaded unique layouts={len(layouts)}")
    by_level = Counter(x["level"] for x in layouts.values())
    print("layouts by level:", dict(sorted(by_level.items())))

    rng = random.Random(args.seed)
    chosen = select_layouts_with_tasks(
        layouts,
        train_n=args.train_layouts,
        val_n=args.val_layouts,
        test_n=args.test_layouts,
        tasks_per_layout=args.tasks_per_layout,
        rng=rng,
    )
    split_layouts_by_name = {
        split: [layout for layout, _tasks in chosen[split]]
        for split in SPLITS
    }
    tasks_by_split = {
        split: [task for _layout, tasks in chosen[split] for task in tasks]
        for split in SPLITS
    }

    # L0: one empty layout, tasks split by task hash.
    l0_layout = next(x for x in layouts.values() if x["level"] == "L0")
    l0_tasks = generate_tasks(
        l0_layout, args.l0_train + args.l0_val + args.l0_test, rng)
    if len(l0_tasks) < args.l0_train + args.l0_val + args.l0_test:
        raise RuntimeError(
            f"L0 generated only {len(l0_tasks)} tasks, need "
            f"{args.l0_train + args.l0_val + args.l0_test}")
    l0_tasks.sort(key=lambda x: x["task_id"])
    tasks_by_split["train"].extend(l0_tasks[:args.l0_train])
    tasks_by_split["val"].extend(
        l0_tasks[args.l0_train:args.l0_train + args.l0_val])
    tasks_by_split["test"].extend(
        l0_tasks[args.l0_train + args.l0_val:])
    # L0 layout must be present in every layout split file.
    for split in SPLITS:
        split_layouts_by_name[split] = list(split_layouts_by_name[split])
        split_layouts_by_name[split].append(l0_layout)

    manifest = {
        "version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/build_layout_tasks.py",
        "split_rule": {
            "L1-L4": "layout-level split; all tasks of a layout stay together",
            "L0": "one empty layout; task-level split",
        },
        "quotas": {
            "layouts": {
                "train": args.train_layouts,
                "val": args.val_layouts,
                "test": args.test_layouts,
                "per_level_L1_L4": True,
            },
            "tasks_per_layout": args.tasks_per_layout,
            "L0_tasks": {
                "train": args.l0_train,
                "val": args.l0_val,
                "test": args.l0_test,
            },
        },
        "files": {},
        "counts": {},
    }

    for split in SPLITS:
        lp = out / "layouts" / f"{split}.jsonl"
        tp = out / "tasks" / f"{split}.jsonl"
        lhash = write_jsonl(lp, split_layouts_by_name[split])
        thash = write_jsonl(tp, tasks_by_split[split])
        manifest["files"][f"layouts/{split}.jsonl"] = {
            "sha256": lhash,
            "count": len(split_layouts_by_name[split]),
        }
        manifest["files"][f"tasks/{split}.jsonl"] = {
            "sha256": thash,
            "count": len(tasks_by_split[split]),
        }
        manifest["counts"][split] = {
            "layouts": len(split_layouts_by_name[split]),
            "tasks": len(tasks_by_split[split]),
            "levels": dict(sorted(Counter(
                t["level"] for t in tasks_by_split[split]).items())),
        }

    mpath = out / "layout_tasks_manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"done {(time.time()-t0):.1f}s")
    for split in SPLITS:
        print(split, manifest["counts"][split])
    print("manifest ->", mpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
