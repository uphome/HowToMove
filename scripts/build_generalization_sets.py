"""Build held-out map sets for zero-shot generalization evaluation.

The training/evaluation dataset is fixed at 12x12.  This script creates
separate evaluation-only datasets, without changing the training data:

* size suites: maps generated at several dimensions, one directory per size;
* same-wall suites: alternate start/goal pairs on held-out test walls.

The output layout is compatible with ``scripts/sft_agent.py``:

    <out>/size_8x8/maps/{val,test}.jsonl
    <out>/same_wall_L4/maps/{val,test}.jsonl

Example:

    python scripts/build_generalization_sets.py \
        --out /tmp/opencode/generalization \
        --sizes 8,10,16,20 \
        --levels L0,L4 \
        --n 100 \
        --same-wall

The agent is still evaluated with the greedy protocol used by SFT/GRPO.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import MapRecord, load_records, save_records  # noqa: E402
from env.maze import LEVELS, LEVEL_RULES, Maze, _place, generate  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def budget_for(width: int, height: int, mode: str) -> int:
    if mode == "fixed":
        return 30
    # Enough room for a path much longer than the Manhattan distance.
    return max(30, 2 * (width + height))


def build_size_suite(
    out: Path,
    *,
    sizes: list[int],
    levels: list[str],
    n: int,
    budget: str,
    seed: int,
) -> dict:
    manifest = {"kind": "size", "budget": budget, "suites": {}}
    for size in sizes:
        width = height = size
        max_steps = budget_for(width, height, budget)
        records: list[MapRecord] = []
        for li, level in enumerate(levels):
            seen: set[str] = set()
            attempt = 0
            level_records: list[MapRecord] = []
            while len(level_records) < n and attempt < max(10000, n * 100):
                try:
                    maze = generate(
                        level,
                        width=width,
                        height=height,
                        max_steps=max_steps,
                        seed=seed + size * 100_003 + li * 1_000_003 + attempt,
                        max_attempts=1000,
                    )
                except RuntimeError:
                    attempt += 1
                    continue
                rec = MapRecord.from_maze(
                    maze,
                    ident=f"GEN-{size}x{size}-{level}-{attempt:06d}",
                    seed=attempt,
                )
                if rec.hash not in seen:
                    seen.add(rec.hash)
                    level_records.append(rec)
                attempt += 1
            if len(level_records) < n:
                raise RuntimeError(
                    f"size {size} level {level}: only {len(level_records)}/{n}"
                )
            records.extend(level_records)
        suite = out / f"size_{size}x{size}" / "maps"
        save_records(records, suite / "test.jsonl")
        save_records([], suite / "val.jsonl")
        manifest["suites"][f"size_{size}x{size}"] = {
            "width": size,
            "height": size,
            "max_steps": max_steps,
            "levels": levels,
            "n_per_level": n,
            "total": len(records),
            "level_counts": dict(Counter(r.level for r in records)),
        }
        print(f"size {size}x{size} -> {suite} ({len(records)} maps)")
    return manifest


def build_same_wall_suite(
    out: Path,
    *,
    levels: list[str],
    n: int,
    seed: int,
    test_data: Path,
) -> dict:
    test_records = load_records(test_data / "maps" / "test.jsonl")
    by_level: dict[str, list[MapRecord]] = {}
    for rec in test_records:
        by_level.setdefault(rec.level, []).append(rec)

    rng = random.Random(seed)
    records: list[MapRecord] = []
    for li, level in enumerate(levels):
        source = sorted(by_level.get(level, []), key=lambda r: r.hash)
        made = 0
        wall_sets = 0
        for rec in source:
            if made >= n:
                break
            rules = LEVEL_RULES[level]
            placed = None
            for _ in range(200):
                candidate = _place(
                    set(rec.walls),
                    rec.width,
                    rec.height,
                    rec.max_steps,
                    rules["min_dist"],
                    rules["min_detour"],
                    rng,
                )
                if candidate is None:
                    break
                start, goal, dist = candidate
                if (start, goal) != (rec.start, rec.goal):
                    placed = (start, goal, dist)
                    break
            if placed is None:
                continue
            start, goal, dist = placed
            maze = Maze(
                width=rec.width,
                height=rec.height,
                walls=rec.walls,
                start=start,
                goal=goal,
                max_steps=rec.max_steps,
                level=rec.level,
                seed=seed + li,
                dist=dist,
            )
            alt = MapRecord.from_maze(
                maze,
                ident=f"ALT-{level}-{made:04d}",
                seed=seed + li,
            )
            records.append(alt)
            made += 1
            wall_sets += 1
        if made < n:
            raise RuntimeError(
                f"same-wall {level}: only {made}/{n} alternate maps")
        print(f"same-wall {level}: {made} maps from {wall_sets} held-out walls")

    suite = out / "same_wall" / "maps"
    save_records(records, suite / "test.jsonl")
    save_records([], suite / "val.jsonl")
    return {
        "kind": "same_wall",
        "levels": levels,
        "n_per_level": n,
        "total": len(records),
        "level_counts": dict(Counter(r.level for r in records)),
        "source": str(test_data / "maps" / "test.jsonl"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/generalization")
    ap.add_argument("--sizes", default="8,10,16,20")
    ap.add_argument("--levels", default="L0,L4")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--budget", choices=("fixed", "scaled"), default="fixed")
    ap.add_argument("--same-wall", action="store_true")
    ap.add_argument("--same-wall-levels", default="L4")
    ap.add_argument("--same-wall-n", type=int, default=None,
                    help="same-wall maps per level; defaults to --n")
    ap.add_argument("--test-data", default="data")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    sizes = [int(x) for x in parse_csv(args.sizes)]
    levels = [x.upper() for x in parse_csv(args.levels)]
    same_levels = [x.upper() for x in parse_csv(args.same_wall_levels)]
    for level in levels + same_levels:
        if level not in LEVELS:
            ap.error(f"unknown level: {level}")
    if args.n < 1:
        ap.error("--n must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "out": str(out),
        "sizes": sizes,
        "levels": levels,
        "n": args.n,
        "budget": args.budget,
        "seed": args.seed,
    }
    manifest["size"] = build_size_suite(
        out,
        sizes=sizes,
        levels=levels,
        n=args.n,
        budget=args.budget,
        seed=args.seed,
    )
    if args.same_wall:
        manifest["same_wall"] = build_same_wall_suite(
            out,
            levels=same_levels,
            n=args.same_wall_n or args.n,
            seed=args.seed,
            test_data=Path(args.test_data),
        )
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
