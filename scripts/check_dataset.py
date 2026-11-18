"""数据集门禁：无泄漏 + 可解性 + 一致性 + manifest 校验。

    python scripts/check_dataset.py
    python scripts/check_dataset.py --data data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    content_hash, load_json, load_records, optimal_actions,
)
from env.maze import INF, LEVELS, manhattan  # noqa: E402
from env.render import render_state  # noqa: E402

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--strict-counts", action="store_true", default=True)
    args = ap.parse_args()
    root = Path(args.data)
    errors: list[str] = []
    warns: list[str] = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    manifest = load_json(root / "manifest.json")

    # ---- 1. manifest 哈希 ----
    for rel, meta in manifest["files"].items():
        p = root / rel
        check(p.exists(), f"缺文件 {rel}")
        if p.exists():
            check(sha256_file(p) == meta["sha256"], f"sha256 不匹配 {rel}")

    # ---- 2. 读入 ----
    splits = {}
    for name in ("train", "val", "test_id", "test_hard", "test_ood"):
        splits[name] = load_records(root / "maps" / f"{name}.jsonl")
        print(f"{name:10s} {len(splits[name]):5d}")

    # ---- 3. 哈希不重叠 ----
    hashes = {name: {r.hash for r in recs} for name, recs in splits.items()}
    names = list(hashes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = hashes[names[i]] & hashes[names[j]]
            check(not inter, f"{names[i]} ∩ {names[j]} = {len(inter)} 张重复地图")

    # ---- 4. 每张图 ----
    for name, recs in splits.items():
        ids = [r.id for r in recs]
        check(len(ids) == len(set(ids)), f"{name} 有重复 id")
        check(len({r.hash for r in recs}) == len(recs), f"{name} 内部有重复地图")
        for r in recs:
            maze = r.to_maze()
            d = maze.distance(maze.start)
            check(d != INF, f"{name}/{r.id} 不可解")
            check(d == r.shortest, f"{name}/{r.id} shortest 不一致 {d} vs {r.shortest}")
            check(d <= r.max_steps - 2,
                  f"{name}/{r.id} 超步数预算 {d} > {r.max_steps - 2}")
            check(r.detour == r.shortest - manhattan(r.start, r.goal),
                  f"{name}/{r.id} detour 不一致")
            check(r.hash == content_hash(r.width, r.height, r.walls,
                                         r.start, r.goal),
                  f"{name}/{r.id} hash 不规范")
            check(r.start not in r.walls and r.goal not in r.walls,
                  f"{name}/{r.id} 起点/终点在墙上")
            check(r.level in LEVELS, f"{name}/{r.id} 非法级别")
            if name == "test_hard":
                check(r.detour >= 6, f"{name}/{r.id} detour < 6")
                check(r.level in ("L2", "L3", "L4"), f"{name}/{r.id} 级别不该进 hard")

    # ---- 5. eval_states（新 Agent 数据只保留地图，可不存在）----
    states = []
    eval_path = root / "eval_states.jsonl"
    if eval_path.exists() and "eval_states.jsonl" in manifest["files"]:
        states = [json.loads(line) for line in
                  eval_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        by_id = {r.id: r for r in splits["test_id"]}
        check(len(states) == manifest["files"]["eval_states.jsonl"]["count"],
              "eval_states 计数与 manifest 不符")
        for s in states:
            rec = by_id.get(s["map_id"])
            check(rec is not None, f"{s['id']} 引用了不存在的 map_id")
            if rec is None:
                continue
            maze = rec.to_maze()
            pos = tuple(s["pos"])
            expect = optimal_actions(maze, pos)
            check(s["optimal_actions"] == expect,
                  f"{s['id']} optimal_actions 不一致 {s['optimal_actions']} vs {expect}")
            check(bool(expect), f"{s['id']} 没有最优动作（终点？）")
            check(s["state"] == render_state(maze, pos), f"{s['id']} state 文本不一致")
            check(maze.distance(pos) >= 3, f"{s['id']} 距离太近")
    elif eval_path.exists():
        warns.append("eval_states.jsonl 存在但不在 manifest 中，已跳过")

    # ---- 6. 配额（以 manifest 为准，支持任意规模）----
    if args.strict_counts:
        quotas = manifest.get("quotas", {})
        for rel, meta in manifest["files"].items():
            got = meta["count"]
            check(got > 0, f"{rel} 计数为 0")
        for split, key in (("train", "train_per_level"),
                           ("val", "val_per_level"),
                           ("test_id", "test_per_level")):
            want = quotas.get(key)
            if want:
                dist = {lv: sum(1 for r in splits[split] if r.level == lv)
                        for lv in LEVELS}
                check(all(v == want for v in dist.values()),
                      f"{split} 分层不均：{dist}，期望每级 {want}")
        if quotas.get("hard"):
            check(len(splits["test_hard"]) == quotas["hard"],
                  f"test_hard {len(splits['test_hard'])} != {quotas['hard']}")
        if quotas.get("ood_per_level"):
            want_ood = quotas["ood_per_level"] * 4 * len(manifest["generator"]["ood_sizes"])
            check(len(splits["test_ood"]) == want_ood,
                  f"test_ood {len(splits['test_ood'])} != {want_ood}")
        if quotas.get("eval_states"):
            check(len(states) == quotas["eval_states"],
                  f"eval_states {len(states)} != {quotas['eval_states']}")

    # ---- 输出 ----
    print()
    for w in warns:
        print(f"WARN  {w}")
    if errors:
        print(f"FAIL  {len(errors)} 项")
        for e in errors[:20]:
            print("  -", e)
        return 1
    print("PASS  数据集检查全部通过")
    print(f"  地图总数 {sum(len(v) for v in splits.values())}")
    print(f"  各 split 哈希互不相交；所有图可解且 shortest <= max_steps-2")
    if states:
        print(f"  eval_states 的 optimal_actions 与 BFS 重算一致")
    else:
        print("  eval_states 未保留（map-only 数据集）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
