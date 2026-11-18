"""Validate layout-level splits and generated tasks."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.maze import INF, LEVEL_RULES, bfs_field, manhattan  # noqa: E402

SPLITS = ("train", "val", "test")


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    errors = []
    manifest = json.loads((root / "layout_tasks_manifest.json").read_text())

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    for rel, meta in manifest["files"].items():
        p = root / rel
        check(p.exists(), f"missing {rel}")
        if p.exists():
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            check(got == meta["sha256"], f"sha256 mismatch {rel}")
            check(len(load(p)) == meta["count"], f"count mismatch {rel}")

    layouts = {s: {r["layout_id"]: r for r in
                   load(root / "layouts" / f"{s}.jsonl")}
               for s in SPLITS}
    tasks = {s: load(root / "tasks" / f"{s}.jsonl") for s in SPLITS}
    for s in SPLITS:
        print(s, "layouts", len(layouts[s]), "tasks", len(tasks[s]))

    for lv in ("L1", "L2", "L3", "L4"):
        sets = {s: {i for i, r in layouts[s].items() if r["level"] == lv}
                for s in SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            check(not (sets[a] & sets[b]),
                  f"{lv} layout overlap {a}/{b}: {len(sets[a] & sets[b])}")

    task_ids = {s: {r["task_id"] for r in tasks[s]} for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(not (task_ids[a] & task_ids[b]),
              f"task overlap {a}/{b}")

    for s in SPLITS:
        layout_map = layouts[s]
        for t in tasks[s]:
            lid = t["layout_id"]
            L = layout_map.get(lid)
            check(L is not None, f"{s}/{t['task_id']} missing layout")
            if L is None:
                continue
            walls = frozenset(tuple(p) for p in L["walls"])
            w, h = L["width"], L["height"]
            start = tuple(t["start"])
            goal = tuple(t["goal"])
            check(start not in walls and goal not in walls,
                  f"{s}/{t['task_id']} start/goal on wall")
            check(0 <= start[0] < w and 0 <= start[1] < h,
                  f"{s}/{t['task_id']} start out of bounds")
            check(0 <= goal[0] < w and 0 <= goal[1] < h,
                  f"{s}/{t['task_id']} goal out of bounds")
            dist = bfs_field(w, h, walls, goal)
            d = dist[start[1]][start[0]]
            det = d - manhattan(start, goal)
            rule = LEVEL_RULES[L["level"]]
            check(d != INF, f"{s}/{t['task_id']} unsolvable")
            check(d == t["shortest"], f"{s}/{t['task_id']} shortest mismatch")
            check(det == t["detour"], f"{s}/{t['task_id']} detour mismatch")
            check(d <= L["max_steps"] - 2, f"{s}/{t['task_id']} too long")
            check(d >= rule["min_dist"], f"{s}/{t['task_id']} too close")
            check(det >= rule["min_detour"], f"{s}/{t['task_id']} detour too low")

    if errors:
        print(f"FAIL {len(errors)}")
        for e in errors[:20]:
            print(" -", e)
        return 1
    print("PASS layout/task checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
