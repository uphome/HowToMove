"""校验动作均衡状态集。

检查项：
  1. manifest 的文件 sha256 / count；
  2. state id / hash / (map_id,pos) 唯一；
  3. map_id 属于对应 split 的地图集；
  4. optimal_actions / action_class / state 文本 / distance 与 BFS 重算一致；
  5. 各 level / split 的 singleton/mixed 配额；
  6. 四个方向的 singleton 数量相等、四个斜对角 mixed 数量相等；
  7. train/val/test 的 map_id 两两不重叠。

    python scripts/check_states.py
    python scripts/check_states.py --states data/states
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    MIXED_CLASSES,
    SINGLETON_CLASSES,
    MapDataset,
    StateDataset,
    action_class,
    optimal_actions,
    state_hash,
)
from env.maze import INF, LEVELS, manhattan  # noqa: E402
from env.render import render_state  # noqa: E402


SPLIT_MAP = {"train": "train", "val": "val", "test": "test_id"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="validate balanced state sets")
    ap.add_argument("--data", default="data")
    ap.add_argument("--states", default="data/states")
    args = ap.parse_args()

    data_dir = Path(args.data)
    states_dir = Path(args.states)
    manifest = json.loads((states_dir / "manifest.json").read_text(
        encoding="utf-8"))
    min_dist = int(manifest["generator"]["min_dist"])
    obs_style = manifest["generator"]["obs_style"]

    errors: list[str] = []
    warnings: list[str] = []
    map_ids_by_split: dict[str, set[str]] = {}

    def fail(msg: str):
        errors.append(msg)

    print(f"manifest: {states_dir/'manifest.json'}  version={manifest.get('version')}")
    for split in ("train", "val", "test"):
        file_info = manifest["files"].get(f"{split}.jsonl")
        if not file_info:
            fail(f"{split}: manifest 缺少文件记录")
            continue
        path = states_dir / f"{split}.jsonl"
        if not path.exists():
            fail(f"{split}: 文件不存在 {path}")
            continue
        if sha256_file(path) != file_info["sha256"]:
            fail(f"{split}: sha256 不匹配")
        ds = StateDataset.load(path)
        if len(ds) != int(file_info["count"]):
            fail(f"{split}: count {len(ds)} != {file_info['count']}")

        map_split = SPLIT_MAP[split]
        maps = {r.id: r for r in MapDataset.load(
            data_dir / "maps" / f"{map_split}.jsonl").records}
        map_ids_by_split[split] = set(maps)
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        seen_pos: set[tuple[str, tuple[int, int]]] = set()
        level_counts: dict[str, Counter] = {}
        level_occ: dict[str, Counter] = {}
        map_ids_used: set[str] = set()

        for rec in ds:
            if rec.id in seen_ids:
                fail(f"{split}: duplicate state id {rec.id}")
            seen_ids.add(rec.id)
            if rec.hash in seen_hashes:
                fail(f"{split}: duplicate state hash {rec.hash}")
            seen_hashes.add(rec.hash)
            key = (rec.map_id, rec.pos)
            if key in seen_pos:
                fail(f"{split}: duplicate (map_id,pos) {key}")
            seen_pos.add(key)

            mrec = maps.get(rec.map_id)
            if mrec is None:
                fail(f"{split}/{rec.id}: map_id {rec.map_id} 不在 {map_split}")
                continue
            if rec.level != mrec.level:
                fail(f"{split}/{rec.id}: level {rec.level} != map {mrec.level}")
            map_ids_used.add(rec.map_id)

            maze = mrec.to_maze()
            if not maze.is_free(*rec.pos):
                fail(f"{split}/{rec.id}: pos {rec.pos} 不是自由格")
                continue
            if rec.pos == maze.goal:
                fail(f"{split}/{rec.id}: pos 在终点上")
            dist = maze.distance(rec.pos)
            if dist >= INF or dist < min_dist:
                fail(f"{split}/{rec.id}: distance {dist} < {min_dist}")
            if dist != rec.distance:
                fail(f"{split}/{rec.id}: distance {rec.distance} != {dist}")
            man = manhattan(rec.pos, maze.goal)
            if man != rec.manhattan:
                fail(f"{split}/{rec.id}: manhattan {rec.manhattan} != {man}")
            acts = tuple(sorted(optimal_actions(maze, rec.pos)))
            if acts != rec.optimal_actions:
                fail(f"{split}/{rec.id}: optimal_actions {rec.optimal_actions} != {acts}")
            cls = action_class(acts)
            if cls != rec.action_class:
                fail(f"{split}/{rec.id}: action_class {rec.action_class} != {cls}")
            text = render_state(maze, rec.pos, style=obs_style)
            if text != rec.state:
                fail(f"{split}/{rec.id}: state 文本与 render_state 不一致")
            h = state_hash(mrec.hash, rec.pos)
            if h != rec.hash:
                fail(f"{split}/{rec.id}: state hash 不一致")

            level_counts.setdefault(rec.level, Counter())[rec.action_class] += 1
            level_occ.setdefault(rec.level, Counter())
            for action in rec.optimal_actions:
                level_occ[rec.level][action] += 1

        if map_ids_used - map_ids_by_split[split]:
            fail(f"{split}: 使用了不属于自己的 map_id")

        # manifest 的 level/class 数量
        for level, counts in manifest["counts"][split].items():
            actual = level_counts.get(level, Counter())
            if sum(actual.values()) != counts["total"]:
                fail(f"{split}/{level}: total {sum(actual.values())} != {counts['total']}")
            for cls, n in counts["actual"].items():
                if actual.get(cls, 0) != n:
                    fail(f"{split}/{level}/{cls}: {actual.get(cls, 0)} != {n}")
            if counts.get("shortfall"):
                fail(f"{split}/{level}: 配额不足 {counts['shortfall']}")
            if level_occ.get(level) and dict(level_occ[level]) != manifest[
                    "action_occurrences"][split][level]:
                fail(f"{split}/{level}: action_occurrences 与 manifest 不一致")

        print(f"{split:5s}: {len(ds):4d} states  "
              f"levels={ {lv: sum(c.values()) for lv, c in sorted(level_counts.items())} }")
        for level in sorted(level_counts):
            actual = level_counts[level]
            s = [actual.get(c, 0) for c in SINGLETON_CLASSES]
            m = [actual.get(c, 0) for c in MIXED_CLASSES]
            if len(set(s)) != 1:
                fail(f"{split}/{level}: singleton 方向不均衡 {s}")
            if len(set(m)) != 1:
                fail(f"{split}/{level}: mixed 组合不均衡 {m}")
            print(f"  {level}: total={sum(actual.values()):3d}  "
                  f"singleton={s}  mixed={m}  "
                  f"occ={dict(sorted(level_occ.get(level, {}).items()))}")

    # cross-split map overlap
    if len(map_ids_by_split) == 3:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = map_ids_by_split[a] & map_ids_by_split[b]
            if overlap:
                fail(f"map_id 跨 split 重叠: {a}/{b}: {sorted(overlap)[:5]}")

    if warnings:
        print("\nWARN")
        for w in warnings:
            print(" ", w)
    if errors:
        print(f"\nFAIL: {len(errors)} 项")
        for e in errors[:50]:
            print(" -", e)
        if len(errors) > 50:
            print(f" ... 还有 {len(errors)-50} 项")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
