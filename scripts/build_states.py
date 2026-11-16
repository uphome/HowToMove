"""构建动作均衡的单步状态集。

不重新生成地图，只从现有 data/maps 里枚举状态：

  map + 任意自由格 pos = 一个单步状态
  optimal_actions = BFS 距离减 1 的动作（可能 1~4 个）

按 optimal_actions 的类别做配额：

  singleton_up / down / left / right   只有一个最优动作
  mixed_dr / dl / ur / ul              两个最优动作（斜对角）

L0 的目标配额：
  train: 每个 singleton 方向 64 + 每个 mixed 组合 16 = 320
  val:   每个 singleton 方向 12 + 每个 mixed 组合 4  = 64
  test:  每个 singleton 方向 12 + 每个 mixed 组合 4  = 64

更高关卡自动降低配额；实际数量写进 manifest。

    python scripts/build_states.py
    python scripts/build_states.py --levels L0 --out data/states
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    MIXED_CLASSES,
    SINGLETON_CLASSES,
    MapDataset,
    StateRecord,
    action_class,
    optimal_actions,
    save_json,
    save_states,
    state_hash,
)
from env.maze import INF, LEVELS, manhattan  # noqa: E402
from env.render import render_state  # noqa: E402


SPLIT_MAP = {"train": "train", "val": "val", "test": "test_id"}

# 每个 level / split 的目标配额。
# singleton = 每个方向的数量；mixed = 每个斜对角组合的数量。
QUOTAS = {
    "L0": {
        "train": {"singleton": 64, "mixed": 16},
        "val": {"singleton": 12, "mixed": 4},
        "test": {"singleton": 12, "mixed": 4},
    },
    "L1": {
        "train": {"singleton": 32, "mixed": 8},
        "val": {"singleton": 8, "mixed": 4},
        "test": {"singleton": 8, "mixed": 4},
    },
    "L2": {
        "train": {"singleton": 32, "mixed": 8},
        "val": {"singleton": 8, "mixed": 4},
        "test": {"singleton": 8, "mixed": 4},
    },
    "L3": {
        "train": {"singleton": 32, "mixed": 8},
        "val": {"singleton": 8, "mixed": 4},
        "test": {"singleton": 8, "mixed": 4},
    },
    "L4": {
        "train": {"singleton": 32, "mixed": 8},
        "val": {"singleton": 8, "mixed": 4},
        "test": {"singleton": 8, "mixed": 4},
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def enumerate_states(records, split: str, level: str, min_dist: int,
                     obs_style: str) -> list[StateRecord]:
    out = []
    for rec in records:
        maze = rec.to_maze()
        for y in range(maze.height):
            for x in range(maze.width):
                pos = (x, y)
                if not maze.is_free(x, y) or pos == maze.goal:
                    continue
                dist = maze.distance(pos)
                if dist >= INF or dist < min_dist:
                    continue
                actions = optimal_actions(maze, pos)
                cls = action_class(actions)
                if cls not in SINGLETON_CLASSES and cls not in MIXED_CLASSES:
                    continue
                h = state_hash(rec.hash, pos)
                out.append(StateRecord(
                    id=f"{split}-{level}-{h[:12]}",
                    map_id=rec.id,
                    level=level,
                    pos=pos,
                    state=render_state(maze, pos, style=obs_style),
                    optimal_actions=tuple(sorted(actions)),
                    action_class=cls,
                    distance=dist,
                    manhattan=manhattan(pos, maze.goal),
                    hash=h,
                ))
    return out


def pick_balanced(candidates: list[StateRecord], split: str,
                  level: str) -> tuple[list[StateRecord], dict]:
    target = QUOTAS[level][split]
    buckets: dict[str, list[StateRecord]] = defaultdict(list)
    for rec in candidates:
        buckets[rec.action_class].append(rec)

    selected: list[StateRecord] = []
    actual: dict[str, int] = {}
    for cls in SINGLETON_CLASSES:
        take = int(target["singleton"])
        picked = sorted(buckets.get(cls, []), key=lambda r: r.hash)[:take]
        selected.extend(picked)
        actual[cls] = len(picked)
    for cls in MIXED_CLASSES:
        take = int(target["mixed"])
        picked = sorted(buckets.get(cls, []), key=lambda r: r.hash)[:take]
        selected.extend(picked)
        actual[cls] = len(picked)

    selected.sort(key=lambda r: (r.level, r.action_class, r.hash))
    shortfall = {
        cls: int(target["singleton"] if cls in SINGLETON_CLASSES else target["mixed"])
        - actual.get(cls, 0)
        for cls in SINGLETON_CLASSES + MIXED_CLASSES
        if actual.get(cls, 0) < (target["singleton"] if cls in SINGLETON_CLASSES
                                 else target["mixed"])
    }
    return selected, {
        "target": target,
        "actual": actual,
        "shortfall": shortfall,
        "candidates": {cls: len(buckets.get(cls, []))
                       for cls in SINGLETON_CLASSES + MIXED_CLASSES},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="build balanced single-step states")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/states")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--min-dist", type=int, default=3)
    ap.add_argument("--obs-style", default="minimal")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    levels = tuple(x.strip().upper() for x in args.levels.split(",") if x.strip())
    splits = tuple(x.strip() for x in args.splits.split(",") if x.strip())
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        ap.error(f"unknown level(s): {bad}")
    bad = [x for x in splits if x not in SPLIT_MAP]
    if bad:
        ap.error(f"unknown split(s): {bad}; expected {tuple(SPLIT_MAP)}")

    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/build_states.py",
        "generator": {
            "source": "data/maps/*.jsonl",
            "min_dist": args.min_dist,
            "obs_style": args.obs_style,
        },
        "quotas": {lv: {sp: QUOTAS[lv][sp] for sp in splits}
                   for lv in levels},
        "files": {},
        "counts": {},
        "action_occurrences": {},
        "totals": {},
    }

    total_all = 0
    for split in splits:
        map_split = SPLIT_MAP[split]
        ds = MapDataset.load(data_dir / "maps" / f"{map_split}.jsonl")
        all_states: list[StateRecord] = []
        split_counts: dict[str, dict] = {}
        split_occ: dict[str, dict] = {}
        print(f"\n=== {split} ({map_split}) ===")
        for level in levels:
            records = [r for r in ds.records if r.level == level]
            if not records:
                print(f"{level}: no maps, skip")
                continue
            t0 = time.time()
            candidates = enumerate_states(
                records, split, level, args.min_dist, args.obs_style)
            picked, info = pick_balanced(candidates, split, level)
            all_states.extend(picked)
            split_counts[level] = {
                "total": len(picked),
                "actual": info["actual"],
                "target": info["target"],
                "shortfall": info["shortfall"],
                "candidates": info["candidates"],
            }
            occ = Counter()
            for rec in picked:
                for action in rec.optimal_actions:
                    occ[action] += 1
            split_occ[level] = dict(sorted(occ.items()))
            single_actual = {k.split("_", 1)[1]: v
                             for k, v in info["actual"].items()
                             if k in SINGLETON_CLASSES}
            mixed_actual = {k.split("_", 1)[1]: v
                            for k, v in info["actual"].items()
                            if k in MIXED_CLASSES}
            print(f"{level}: {len(picked):3d} states  "
                  f"singleton={single_actual}  mixed={mixed_actual}  "
                  f"({time.time()-t0:.1f}s)")
            if info["shortfall"]:
                print(f"  shortfall: {info['shortfall']}")

        path = out_dir / f"{split}.jsonl"
        n = save_states(all_states, path)
        total_all += n
        manifest["files"][f"{split}.jsonl"] = {
            "count": n,
            "sha256": sha256_file(path),
        }
        manifest["counts"][split] = split_counts
        manifest["action_occurrences"][split] = split_occ
        manifest["totals"][split] = n

    manifest["totals"]["all"] = total_all
    save_json(manifest, out_dir / "manifest.json")
    print(f"\n写入 {out_dir}/  states={total_all}  manifest={out_dir/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
