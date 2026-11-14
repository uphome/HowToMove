"""在数据集上评估真模型（不做任何训练）。

    # 只看数字
    python scripts/eval_llm.py --split test_id --limit 10

    # 同时推到浏览器看轨迹
    python -m env.ui --port 7861
    python scripts/eval_llm.py --split test_id --limit 5 --ui-url http://127.0.0.1:7861

指标：成功率 / 最优性（步数÷最短路）/ 撞墙 / 格式错 / 总奖励 / 结束原因
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from dataset import MapDataset  # noqa: E402
from engine_policy import EnginePolicy  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.ui import VizClient  # noqa: E402
from env.maze import LEVELS  # noqa: E402
from rollout import run_episode  # noqa: E402


def pick_balanced(ds: MapDataset, limit: int) -> list:
    """按级别轮转取，保证覆盖 L0-L4。"""
    pools = {lv: [r for r in ds.records if r.level == lv] for lv in LEVELS}
    per = max(1, math.ceil(limit / len(LEVELS)))
    out = []
    for i in range(per):
        for lv in LEVELS:
            if i < len(pools[lv]) and len(out) < limit:
                out.append(pools[lv][i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test_id",
                    choices=["test_id", "test_hard", "val", "train", "test_ood"])
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--obs-style", default="minimal")
    ap.add_argument("--memory", type=int, default=0)
    ap.add_argument("--ui-url", default=None)
    ap.add_argument("--json", default="out/eval_llm.json")
    ap.add_argument("--max-batch", type=int, default=1)
    args = ap.parse_args()

    ds = MapDataset.load(Path(args.data) / "maps" / f"{args.split}.jsonl")
    records = pick_balanced(ds, args.limit)
    print(f"split={args.split}  maps={len(records)}  "
          f"levels={sorted({r.level for r in records})}  T={args.temperature}")

    ui = None
    if args.ui_url:
        client = VizClient(args.ui_url)
        if client.health():
            ui = client
            print(f"UI: {args.ui_url}")
        else:
            print(f"UI: {args.ui_url} 不可用，只输出数字")

    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    engine = RolloutEngine(model, tokenizer, max_batch=args.max_batch,
                           max_seq_len=768, kv_ratio=0.5, device=device)
    policy = EnginePolicy(engine, tokenizer, Sampling(
        max_tokens=48, temperature=args.temperature))

    rows = []
    t0 = time.time()
    try:
        for i, rec in enumerate(records, 1):
            maze = rec.to_maze()
            tag = f"{args.split}-{i}"
            traj = run_episode(maze, policy, ui=ui, episode=tag,
                               obs_style=args.obs_style, memory=args.memory)
            collisions = sum(1 for t in traj.turns
                             if t.info.get("reason") == "collision")
            fmterr = sum(1 for t in traj.turns if t.action == "invalid")
            reached = traj.success
            rows.append({
                "id": rec.id, "level": rec.level,
                "shortest": rec.shortest, "detour": rec.detour,
                "steps": traj.steps, "success": reached,
                "truncated": traj.truncated, "reward": round(traj.total_reward, 2),
                "collisions": collisions, "format_errors": fmterr,
                "optimality": (round(traj.steps / rec.shortest, 3)
                               if reached else None),
                "final_dist": maze.distance(traj.turns[-1].pos) if traj.turns
                              else maze.distance(maze.start),
            })
            r = rows[-1]
            print(f"[{i}/{len(records)}] {rec.level} {rec.id:16s} "
                  f"shortest={rec.shortest:2d} steps={r['steps']:2d} "
                  f"success={str(reached):5s} coll={collisions:2d} "
                  f"fmt={fmterr} rew={r['reward']:7.2f} "
                  f"dist_left={r['final_dist']:2d}")
    finally:
        engine.close()

    dt = time.time() - t0
    n = len(rows)
    succ = sum(r["success"] for r in rows)
    print("\n" + "=" * 78)
    print(f"总计 {n} 张  成功率 {succ}/{n} = {succ/n:.0%}   "
          f"耗时 {dt/60:.1f} min   平均 {dt/n:.1f}s/局")
    print(f"平均撞墙 {statistics.mean(r['collisions'] for r in rows):.1f} 次/局，"
          f"格式错 {sum(r['format_errors'] for r in rows)} 次，"
          f"平均奖励 {statistics.mean(r['reward'] for r in rows):+.2f}")
    print("-" * 78)
    print(f"{'级别':6s} {'局数':>4s} {'成功':>4s} {'平均步数':>8s} "
          f"{'平均奖励':>9s} {'平均撞墙':>8s} {'平均剩余距离':>12s}")
    for lv in LEVELS:
        sub = [r for r in rows if r["level"] == lv]
        if not sub:
            continue
        print(f"{lv:6s} {len(sub):>4d} {sum(r['success'] for r in sub):>4d} "
              f"{statistics.mean(r['steps'] for r in sub):>8.1f} "
              f"{statistics.mean(r['reward'] for r in sub):>+9.2f} "
              f"{statistics.mean(r['collisions'] for r in sub):>8.1f} "
              f"{statistics.mean(r['final_dist'] for r in sub):>12.1f}")
    print("=" * 78)
    ok = [r for r in rows if r["success"]]
    if ok:
        print(f"成功局的平均最优性（步数/最短路）："
              f"{statistics.mean(r['optimality'] for r in ok):.2f}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({
        "split": args.split, "temperature": args.temperature,
        "obs_style": args.obs_style, "model": args.model,
        "elapsed_s": round(dt, 1), "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
