"""多步 GRPO 训练 + 实时 UI 可视化。

    # 终端 A：UI（或不启，脚本自己起）
    python -m env.ui --port 7861

    # 终端 B：训练，每步动作推到 UI
    python scripts/train_grpo.py --updates 5 --maps-per-update 1 --g 8 \
        --ui-url http://127.0.0.1:7861

每次更新：采样 K 张图 → 每张 G 条同步轨迹 → 组内标准化 → GRPO 更新。
浏览器里可以：
  - 「训练看板」：最新指标、loss/reward/success/collision/ratio/gnorm 曲线、
    最近 30 次更新历史；
  - 「推理轨迹」：实时迷宫、逐 turn 的 action/reward/logprob/tokens/latency，
    点击任一步回看该步的模型原始输出和 prompt；
  - 轨迹下拉里切换任意 episode（暂停跟随），用「跟随最新」恢复实时。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

import torch  # noqa: E402

from dataset import MapDataset  # noqa: E402
from engine_policy import EnginePolicy  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.ui import TrainViz  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.train import (  # noqa: E402
    GRPOTrainer, TrainConfig, group_advantages, samples_from_trajectory,
)
from rollout import run_group  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--maps-per-update", type=int, default=1)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--obs-style", default="minimal")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ui-url", default=None)
    ap.add_argument("--ui-host", default="127.0.0.1",
                    help="自带 UI 时监听地址；远程访问可设 0.0.0.0")
    ap.add_argument("--ui-port", type=int, default=7862)
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--json", default="out/train_grpo.json")
    ap.add_argument("--save", default="out/lora_grpo.pt")
    args = ap.parse_args()

    ds = MapDataset.load(Path(args.data) / "maps" / f"{args.split}.jsonl")
    rng = random.Random(args.seed)

    viz = None
    if not args.no_ui:
        viz = TrainViz(args.ui_url, host=args.ui_host, port=args.ui_port,
                       serve=args.ui_url is None)
        print(f"UI: {viz.url}")

    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    tie_embeddings(model)

    engine = RolloutEngine(model, tokenizer, max_batch=args.g,
                           max_seq_len=768, kv_ratio=0.5, device=device)
    policy = EnginePolicy(engine, tokenizer, Sampling(
        max_tokens=args.max_tokens, temperature=args.temperature))
    adapter = LoRAAdapter(model, r=args.lora_r, alpha=2 * args.lora_r)
    trainer = GRPOTrainer(adapter, TrainConfig(lr=args.lr, clip_eps=args.clip,
                                               beta=args.beta))
    print(f"LoRA {sum(p.numel() for p in adapter.trainable_parameters())/1e6:.2f}M  "
          f"KV 池 {engine.pool_stats}")

    hist = []
    t_start = time.time()
    for u in range(1, args.updates + 1):
        maps = ds.sample(args.maps_per_update, rng, levels=("L1", "L2", "L3", "L4"))
        samples = []
        returns, successes, steps_used, collisions, fmterr = [], 0, [], 0, 0
        degenerate = 0
        for mi, rec in enumerate(maps):
            maze = rec.to_maze()
            trajs = run_group(maze, policy, args.g, ui=viz,
                              episode_prefix=f"u{u}m{mi}", obs_style=args.obs_style)
            group_returns = [t.total_reward for t in trajs]
            advs, degen = group_advantages(group_returns)
            returns.extend(group_returns)
            if degen:
                degenerate += 1
            else:
                for traj, adv in zip(trajs, advs):
                    samples.extend(samples_from_trajectory(traj, adv))
            for t in trajs:
                successes += int(t.success)
                steps_used.append(t.steps)
                collisions += sum(1 for x in t.turns
                                  if x.info.get("reason") == "collision")
                fmterr += sum(1 for x in t.turns if x.action == "invalid")

        stats = None
        if samples:
            stats = trainer.update(samples)

        rec_metrics = {
            "update": u,
            "total_updates": args.updates,
            "episodes": len(returns),
            "mean_reward": round(statistics.mean(returns), 2),
            "success": successes,
            "mean_steps": round(statistics.mean(steps_used), 1),
            "collisions": collisions,
            "format_errors": fmterr,
            "degenerate": degenerate,
            "samples": len(samples),
        }
        if stats is not None:
            rec_metrics.update({
                "loss": round(stats.loss, 5),
                "pg": round(stats.pg, 5),
                "kl": round(stats.kl, 5),
                "ratio_min": round(stats.ratio_min, 4),
                "ratio_max": round(stats.ratio_max, 4),
                "clip_frac": round(stats.clip_frac, 3),
                "drop_frac": round(stats.drop_frac, 3),
                "grad_norm": round(stats.grad_norm, 4),
            })
        hist.append(rec_metrics)
        print(f"[{u}/{args.updates}] " + "  ".join(
            f"{k}={v}" for k, v in rec_metrics.items() if k != "update"))
        if viz is not None:
            viz.update(**rec_metrics)

    dt = time.time() - t_start
    print(f"\n总耗时 {dt/60:.1f} min")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.save)
    Path(args.json).write_text(json.dumps(hist, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(f"adapter → {args.save}\n日志 → {args.json}")

    engine.close()
    if viz is not None:
        viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
