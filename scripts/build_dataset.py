"""构建 train/val/test 地图集 + 固定单步评测状态。

    python scripts/build_dataset.py
    python scripts/build_dataset.py --seeds 3000 --out data_small

流程：生成 → 内容哈希去重 → 哈希分桶 → 配额落盘 → manifest(sha256)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    MapRecord, build_eval_states, save_json, save_records, split_for_hash,
)
from env.maze import LEVEL_RULES, LEVELS, generate  # noqa: E402

MAIN_SIZE = 12
HARD_LEVELS = ("L2", "L3", "L4")
OOD_SIZES = ((8, 8), (16, 16))


def gen_candidates(seeds: int, width: int, height: int, max_steps: int,
                   levels=LEVELS) -> tuple[dict[str, MapRecord], int]:
    """按 seed 生成并去重，返回 {hash: record} 与失败数。"""
    out: dict[str, MapRecord] = {}
    failed = 0
    for level in levels:
        for seed in range(seeds):
            try:
                maze = generate(level=level, width=width, height=height,
                                max_steps=max_steps, seed=seed)
            except RuntimeError:
                failed += 1
                continue
            rec = MapRecord.from_maze(
                maze, ident=f"{level}-{seed:06d}", seed=seed)
            out.setdefault(rec.hash, rec)
    return out, failed


def by_level(records) -> dict[str, list[MapRecord]]:
    out: dict[str, list[MapRecord]] = defaultdict(list)
    for r in records:
        out[r.level].append(r)
    return out


def take(pool: list[MapRecord], k: int) -> list[MapRecord]:
    return sorted(pool, key=lambda r: r.hash)[:k]


def take_balanced(pools: dict[str, list[MapRecord]], k: int) -> list[MapRecord]:
    """按级别轮转取，避免 hard 集全被某一级占满。"""
    levels = sorted(pools)
    ordered = {lv: sorted(pools[lv], key=lambda r: r.hash) for lv in levels}
    out: list[MapRecord] = []
    i = 0
    while len(out) < k:
        progressed = False
        for lv in levels:
            if i < len(ordered[lv]) and len(out) < k:
                out.append(ordered[lv][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seeds", type=int, default=6000,
                    help="每个级别的主集候选 seed 数")
    ap.add_argument("--ood-seeds", type=int, default=2500,
                    help="每个尺寸/级别的 OOD 候选 seed 数")
    ap.add_argument("--train-per-level", type=int, default=600)
    ap.add_argument("--val-per-level", type=int, default=80)
    ap.add_argument("--test-per-level", type=int, default=80)
    ap.add_argument("--hard", type=int, default=200)
    ap.add_argument("--ood-per-level", type=int, default=25)
    ap.add_argument("--eval-states", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out)
    t0 = time.time()

    print(f"[1/5] 生成候选：{args.seeds} seeds × {len(LEVELS)} levels "
          f"（{MAIN_SIZE}x{MAIN_SIZE}）")
    cands, failed = gen_candidates(args.seeds, MAIN_SIZE, MAIN_SIZE,
                                   args.max_steps)
    print(f"      唯一地图 {len(cands)}，失败 {failed}")

    print("[2/5] 按内容哈希分桶")
    buckets: dict[str, list[MapRecord]] = defaultdict(list)
    for rec in cands.values():
        buckets[split_for_hash(rec.hash)].append(rec)
    for name in ("train", "val", "test_id", "test_hard"):
        print(f"      {name:10s} {len(buckets[name]):5d}")

    print("[3/5] 配额 + 分层")
    quotas = {
        "train": args.train_per_level,
        "val": args.val_per_level,
        "test_id": args.test_per_level,
    }
    selected: dict[str, list[MapRecord]] = {}
    for split, k in quotas.items():
        pools = by_level(buckets[split])
        picked = []
        for level in LEVELS:
            got = take(pools.get(level, []), k)
            if len(got) < k:
                print(f"      !! {split}/{level} 不足：{len(got)} < {k}")
            picked.extend(got)
        selected[split] = picked
        print(f"      {split:10s} {len(picked):5d}")

    hard_pools = {
        lv: [r for r in buckets["test_hard"] if r.level == lv and r.detour >= 6]
        for lv in HARD_LEVELS
    }
    hard = take_balanced(hard_pools, args.hard)
    if len(hard) < args.hard:
        print(f"      !! test_hard 不足：{len(hard)} < {args.hard}")
    selected["test_hard"] = hard
    hard_by_level = {lv: sum(1 for r in hard if r.level == lv) for lv in HARD_LEVELS}
    print(f"      test_hard  {len(hard):5d}  {hard_by_level}")

    print("[4/5] OOD 测试集（8x8 / 16x16）")
    ood: list[MapRecord] = []
    seen = set(cands)
    for w, h in OOD_SIZES:
        items, bad = gen_candidates(args.ood_seeds, w, h, args.max_steps,
                                    levels=("L1", "L2", "L3", "L4"))
        pools = {lv: [] for lv in ("L1", "L2", "L3", "L4")}
        for rec in items.values():
            if rec.hash in seen or split_for_hash(rec.hash) != "test_hard":
                continue
            seen.add(rec.hash)
            pools[rec.level].append(rec)
        picked = take_balanced(pools, args.ood_per_level * 4)
        for rec in picked:
            ood.append(MapRecord(**{**rec.__dict__,
                                    "id": f"ood{w}-{rec.level}-{rec.hash[:8]}"}))
        print(f"      {w}x{h}: {len(picked)}（失败 {bad}）")
    selected["test_ood"] = ood

    print("[5/5] 写盘 + 单步评测状态")
    files = {}
    for split, recs in selected.items():
        path = out / "maps" / f"{split}.jsonl"
        n = save_records(recs, path)
        files[path.relative_to(out).as_posix()] = {
            "count": n, "sha256": sha256_file(path)}

    eval_records = []
    per_level = max(1, args.eval_states // len(LEVELS))
    for level in LEVELS:
        eval_records.extend(take(by_level(selected["test_id"]).get(level, []),
                                 per_level))
    states = build_eval_states(eval_records, len(eval_records))
    path = out / "eval_states.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False,
                                         separators=(",", ":"))
                              for s in states) + "\n", encoding="utf-8")
    files[path.relative_to(out).as_posix()] = {
        "count": len(states), "sha256": sha256_file(path)}

    manifest = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "builder": "scripts/build_dataset.py",
        "generator": {
            "file": "env/maze.py",
            "level_rules": LEVEL_RULES,
            "max_steps": args.max_steps,
            "main_size": [MAIN_SIZE, MAIN_SIZE],
            "ood_sizes": [list(s) for s in OOD_SIZES],
        },
        "split_rule": "sha256(canonical)[:8] % 100 -> "
                      "train:<70 val:<84 test_id:<92 test_hard:<100",
        "seeds": {"main": args.seeds, "ood": args.ood_seeds},
        "quotas": {
            "train_per_level": args.train_per_level,
            "val_per_level": args.val_per_level,
            "test_per_level": args.test_per_level,
            "hard": args.hard,
            "ood_per_level": args.ood_per_level,
            "eval_states": args.eval_states,
        },
        "files": files,
        "counts": {k: v["count"] for k, v in files.items()},
    }
    save_json(manifest, out / "manifest.json")

    dt = time.time() - t0
    print(f"\n完成 {dt:.1f}s → {out}/")
    for k, v in files.items():
        print(f"  {k:22s} {v['count']:5d}  {v['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
