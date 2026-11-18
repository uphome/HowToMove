"""Validate the map-only v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import MapRecord, content_hash  # noqa: E402
from env.maze import INF, LEVEL_RULES, LEVELS, bfs_field, manhattan  # noqa: E402

SPLITS = ("train", "val", "test")


def wall_set_key(rec: MapRecord) -> tuple:
    return (rec.width, rec.height, tuple(sorted(rec.walls)))


def load_jsonl(path: Path) -> list[MapRecord]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(MapRecord.from_dict(json.loads(line)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/opencode/maps_v3")
    args = ap.parse_args()

    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text())
    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    records = {split: load_jsonl(root / "maps" / f"{split}.jsonl")
               for split in SPLITS}

    for split in SPLITS:
        path = root / "maps" / f"{split}.jsonl"
        meta = manifest["files"].get(f"maps/{split}.jsonl")
        check(meta is not None, f"manifest missing maps/{split}.jsonl")
        if meta:
            check(meta["count"] == len(records[split]),
                  f"{split}: manifest count mismatch")
            check(meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(),
                  f"{split}: sha256 mismatch")
        check(manifest["counts"].get(split) == len(records[split]),
              f"{split}: counts mismatch")
        check(manifest["levels"].get(split) == dict(Counter(
            r.level for r in records[split])),
              f"{split}: level counts mismatch")

    all_ids = [r.id for rs in records.values() for r in rs]
    all_hashes = [r.hash for rs in records.values() for r in rs]
    check(len(all_ids) == len(set(all_ids)), "duplicate map id")
    check(len(all_hashes) == len(set(all_hashes)), "duplicate map content hash")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = {r.hash for r in records[a]} & {r.hash for r in records[b]}
        check(not overlap, f"map overlap {a}/{b}: {len(overlap)}")

    # L1-L4 split by exact wall set; L0 has one empty wall set and is
    # intentionally map/start-goal split.
    wall_sets = {
        split: {
            level: {wall_set_key(r) for r in rs if r.level == level}
            for level in LEVELS
        }
        for split, rs in records.items()
    }
    for level in ("L1", "L2", "L3", "L4"):
        for a, b in (("train", "val"), ("train", "test"),
                     ("val", "test")):
            overlap = wall_sets[a][level] & wall_sets[b][level]
            check(not overlap,
                  f"{level} wall-set overlap {a}/{b}: {len(overlap)}")

    for split, rs in records.items():
        for rec in rs:
            check(rec.hash == content_hash(
                rec.width, rec.height, rec.walls, rec.start, rec.goal),
                f"{split}/{rec.id}: content hash mismatch")
            check(0 <= rec.start[0] < rec.width and
                  0 <= rec.start[1] < rec.height,
                  f"{split}/{rec.id}: start out of bounds")
            check(0 <= rec.goal[0] < rec.width and
                  0 <= rec.goal[1] < rec.height,
                  f"{split}/{rec.id}: goal out of bounds")
            check(rec.start not in rec.walls and rec.goal not in rec.walls,
                  f"{split}/{rec.id}: start/goal on wall")
            check(all(0 <= x < rec.width and 0 <= y < rec.height
                      for x, y in rec.walls),
                  f"{split}/{rec.id}: wall out of bounds")
            check(rec.level in LEVELS, f"{split}/{rec.id}: bad level")
            check(rec.max_steps >= 4, f"{split}/{rec.id}: bad max_steps")

            dist = bfs_field(rec.width, rec.height, rec.walls, rec.goal)
            d = dist[rec.start[1]][rec.start[0]]
            det = d - manhattan(rec.start, rec.goal)
            check(d != INF, f"{split}/{rec.id}: unsolvable")
            check(d == rec.shortest, f"{split}/{rec.id}: shortest mismatch")
            check(det == rec.detour, f"{split}/{rec.id}: detour mismatch")
            check(d <= rec.max_steps - 2, f"{split}/{rec.id}: too long")
            rule = LEVEL_RULES[rec.level]
            check(d >= rule["min_dist"], f"{split}/{rec.id}: too close")
            check(det >= rule["min_detour"],
                  f"{split}/{rec.id}: detour too low")

    for split in SPLITS:
        print(split, "maps", len(records[split]),
              "levels", dict(sorted(Counter(
                  r.level for r in records[split]).items())),
              "wall_sets", {
                  level: len(wall_sets[split][level])
                  for level in LEVELS
              })

    if errors:
        print(f"FAIL {len(errors)}")
        for msg in errors[:30]:
            print(" -", msg)
        return 1
    print("PASS map dataset checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
