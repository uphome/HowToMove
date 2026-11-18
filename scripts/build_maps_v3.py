"""Generate a map-only dataset for the agent.

The canonical unit is a concrete map:

    map_id + level + walls + start + goal + shortest + detour

Trajectories and steps are generated later from a map. Wall shape is only
used by the generator and level checks; it is not a separate data layer.

Example:

    python scripts/build_maps_v3.py \
      --out /tmp/opencode/maps_v3 \
      --seed 7 \
      --per-level 1000 \
      --train 800 --val 100 --test 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import MapRecord  # noqa: E402
from env.maze import LEVELS, generate  # noqa: E402


def canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, records: list[MapRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(
                rec.to_dict(), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ) + "\n")


def wall_set_key(rec: MapRecord) -> tuple:
    """Concrete wall placement; start/goal are deliberately excluded."""
    return (rec.width, rec.height, tuple(sorted(rec.walls)))


def wall_set_hash(rec: MapRecord) -> str:
    payload = {
        "w": rec.width,
        "h": rec.height,
        "walls": sorted([x, y] for x, y in rec.walls),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def generate_level(level: str, count: int, *, seed: int,
                   width: int, height: int, max_steps: int,
                   max_attempts: int) -> list[MapRecord]:
    # L0 has only one empty wall set, so its split is map/start-goal level.
    # L1-L4 generate one map per concrete wall set. This makes the wall-set
    # split exact and prevents the same walls from appearing in train and
    # val/test with a different start/goal.
    unique_by_wall = level != "L0"
    pool: dict[tuple | str, MapRecord] = {}
    attempt = 0
    while len(pool) < count and attempt < max_attempts:
        try:
            maze = generate(
                level,
                width=width,
                height=height,
                max_steps=max_steps,
                seed=seed + attempt,
            )
        except RuntimeError:
            attempt += 1
            continue
        rec = MapRecord.from_maze(maze)
        key = wall_set_key(rec) if unique_by_wall else rec.hash
        pool.setdefault(key, rec)
        attempt += 1

    if len(pool) < count:
        raise RuntimeError(
            f"{level}: generated only {len(pool)} unique "
            f"{'wall sets' if unique_by_wall else 'maps'} "
            f"after {attempt} attempts, need {count}"
        )
    if unique_by_wall:
        keys = [wall_set_key(r) for r in pool.values()]
        assert len(keys) == len(set(keys)), "duplicate wall set"
    return sorted(pool.values(), key=lambda r: r.hash)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/maps_v3")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-level", type=int, default=1000)
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--height", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-attempts", type=int, default=20000)
    args = ap.parse_args()

    if args.train + args.val + args.test != args.per_level:
        ap.error("train + val + test must equal per-level")
    if args.per_level < 1:
        ap.error("per-level must be positive")

    out = Path(args.out)
    maps = {level: generate_level(
        level,
        args.per_level,
        seed=args.seed + i * 1_000_003,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        max_attempts=args.max_attempts,
    ) for i, level in enumerate(LEVELS)}

    split = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }
    rows = {name: [] for name in split}
    for level in LEVELS:
        records = maps[level]
        if level == "L0":
            # One empty wall set: split maps by start/goal hash.
            ordered = sorted(records, key=lambda r: r.hash)
        else:
            # One map per wall set; sort by the wall-set hash so the split
            # is explicitly wall-set based and reproducible.
            ordered = sorted(records, key=wall_set_hash)
        cursor = 0
        for name, n in split.items():
            rows[name].extend(ordered[cursor:cursor + n])
            cursor += n
        if cursor != len(ordered):
            raise RuntimeError(f"{level}: split does not consume all maps")

    manifest = {
        "version": 3,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/build_maps_v3.py",
        "seed": args.seed,
        "dimensions": {
            "width": args.width,
            "height": args.height,
            "max_steps": args.max_steps,
        },
        "unit": "map",
        "split_rule": {
            "L0": "map-level (one empty wall set; split by start/goal)",
            "L1-L4": (
                "wall-set-level; maps sharing the same exact absolute walls "
                "stay in one split"
            ),
        },
        "wall_set_rule": (
            "wall set = (width, height, sorted absolute walls); "
            "start/goal excluded"
        ),
        "quotas": {
            "per_level": args.per_level,
            "splits": split,
        },
        "counts": {},
        "levels": {},
        "wall_sets": {},
        "files": {},
    }

    for name, records in rows.items():
        path = out / "maps" / f"{name}.jsonl"
        write_jsonl(path, records)
        manifest["files"][f"maps/{name}.jsonl"] = {
            "count": len(records),
            "sha256": canonical_sha(path),
        }
        manifest["counts"][name] = len(records)
        manifest["levels"][name] = dict(sorted(Counter(
            r.level for r in records).items()))
        manifest["wall_sets"][name] = {
            level: len({wall_set_key(r) for r in records
                        if r.level == level})
            for level in LEVELS
        }

    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"maps -> {out}")
    print(f"counts -> {manifest['counts']}")
    print(f"levels -> {manifest['levels']}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
