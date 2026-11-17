"""多步 GRPO 训练 + 固定 val/test 评估 + 实时 UI 可视化。

    # 终端 A：UI（或不启，脚本自己起）
    python -m env.ui --port 7861

    # 终端 B：训练，每步动作推到 UI
    python scripts/train_grpo.py --levels L1 --obs-style rel --fewshot rel_l1 \
        --updates 5 --maps-per-update 1 --g 8 \
        --ui-url http://127.0.0.1:7861

每次更新：采样 K 张图 → 每张 G 条同步轨迹 → 组内标准化 → GRPO 更新。
默认在 update 0 和每 eval-every 次做固定 val 评估，最后做 test 评估。
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

import torch  # noqa: E402

from dataset import MapDataset  # noqa: E402
from engine_policy import EnginePolicy  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import ACTIONS, LEVELS  # noqa: E402
from env.ui import TrainViz  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.multistep import MultiStepGRPOTrainer  # noqa: E402
from grpo.train import (  # noqa: E402
    TrainConfig, group_advantages, samples_from_trajectory,
)
from rollout import run_episode, run_group, run_maps_batch  # noqa: E402


def parse_levels(raw: str) -> tuple[str, ...]:
    levels = tuple(x.strip().upper() for x in raw.split(",") if x.strip())
    if not levels:
        raise ValueError("--levels 不能为空")
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        raise ValueError(f"未知 level: {bad}; 可选 {LEVELS}")
    return levels


def _cuda_cleanup() -> None:
    """释放 eval/rollout 与训练交错产生的缓存碎片。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_eval_maps(ds: MapDataset, levels: tuple[str, ...],
                   per_level: int) -> list:
    """固定评估图：每级按 hash 排序取前 per_level 张。"""
    if per_level <= 0:
        return []
    out = []
    for level in levels:
        pool = sorted((r for r in ds.records if r.level == level),
                      key=lambda r: r.hash)
        out.extend(pool[:per_level])
    return out


def _mean(values) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def summarize_eval(rows: list[dict]) -> dict:
    n = len(rows)
    success = sum(1 for r in rows if r["success"])
    action_counts = Counter()
    for r in rows:
        action_counts.update(r["action_counts"])
    per_level = {}
    for level in sorted({r["level"] for r in rows}):
        sub = [r for r in rows if r["level"] == level]
        ok = [r for r in sub if r["success"]]
        acts = Counter()
        for r in sub:
            acts.update(r["action_counts"])
        per_level[level] = {
            "n": len(sub),
            "success": len(ok),
            "success_rate": round(len(ok) / len(sub), 4) if sub else 0.0,
            "mean_steps": _mean([r["steps"] for r in sub]),
            "mean_reward": _mean([r["reward"] for r in sub]),
            "mean_collisions": _mean([r["collisions"] for r in sub]),
            "format_errors": sum(r["format_errors"] for r in sub),
            "revisits": sum(r["revisits"] for r in sub),
            "action_counts": dict(acts),
            "mean_final_dist": _mean([r["final_dist"] for r in sub]),
            "optimality": _mean([r["optimality"] for r in ok]),
        }
    return {
        "n": n,
        "success": success,
        "success_rate": round(success / n, 4) if n else 0.0,
        "mean_steps": _mean([r["steps"] for r in rows]),
        "mean_reward": _mean([r["reward"] for r in rows]),
        "mean_collisions": _mean([r["collisions"] for r in rows]),
        "format_errors": sum(r["format_errors"] for r in rows),
        "revisits": sum(r["revisits"] for r in rows),
        "action_counts": dict(action_counts),
        "mean_final_dist": _mean([r["final_dist"] for r in rows]),
        "optimality": _mean([r["optimality"] for r in rows
                             if r["optimality"] is not None]),
        "per_level": per_level,
    }


def evaluate_maps(policy: EnginePolicy, maps: list, *,
                  obs_style: str, memory: int, show_visited: bool = False,
                  revisit_penalty: float = 0.0, batch_size: int = 1,
                  split: str = "val") -> tuple[dict, list[dict]]:
    """在固定地图上跑 greedy rollout，返回汇总和逐图结果。"""
    mazes = [rec.to_maze() for rec in maps]
    if batch_size > 1:
        trajs = []
        for start in range(0, len(mazes), batch_size):
            chunk = mazes[start:start + batch_size]
            trajs.extend(run_maps_batch(
                chunk, policy, obs_style=obs_style, memory=memory,
                show_visited=show_visited, revisit_penalty=revisit_penalty,
                episode_prefix=f"eval-{split}-{start // batch_size}",
            ))
    else:
        trajs = [
            run_episode(
                maze, policy, episode=f"eval-{split}-{i}",
                obs_style=obs_style, memory=memory,
                show_visited=show_visited, revisit_penalty=revisit_penalty,
            )
            for i, maze in enumerate(mazes, 1)
        ]

    rows = []
    for i, (rec, maze, traj) in enumerate(zip(maps, mazes, trajs), 1):
        collisions = sum(1 for t in traj.turns
                         if t.info.get("reason") == "collision")
        fmterr = sum(1 for t in traj.turns if t.action == "invalid")
        revisits = sum(1 for t in traj.turns if t.info.get("revisit"))
        action_counts = Counter(t.action for t in traj.turns
                                if t.action in ACTIONS)
        final_pos = traj.turns[-1].pos if traj.turns else maze.start
        rows.append({
            "id": rec.id,
            "level": rec.level,
            "shortest": rec.shortest,
            "detour": rec.detour,
            "steps": traj.steps,
            "success": bool(traj.success),
            "truncated": bool(traj.truncated),
            "reward": round(traj.total_reward, 4),
            "collisions": collisions,
            "format_errors": fmterr,
            "revisits": revisits,
            "action_counts": dict(action_counts),
            "optimality": (round(traj.steps / rec.shortest, 4)
                           if traj.success else None),
            "final_dist": maze.distance(final_pos),
        })
        r = rows[-1]
        print(f"  [{split} {i}/{len(maps)}] {rec.level} {rec.id:16s} "
              f"steps={r['steps']:2d} success={str(r['success']):5s} "
              f"coll={collisions:2d} fmt={fmterr} rev={revisits} "
              f"dist_left={r['final_dist']:2d} reward={r['reward']:7.2f} "
              f"actions={dict(action_counts)}")
    summary = summarize_eval(rows)
    print(f"  [{split}] success={summary['success']}/{summary['n']} "
          f"({summary['success_rate']:.1%}) "
          f"steps={summary['mean_steps']:.1f} "
          f"coll={summary['mean_collisions']:.1f} "
          f"fmt={summary['format_errors']} "
          f"rev={summary['revisits']} "
          f"actions={summary['action_counts']} "
          f"final_dist={summary['mean_final_dist']:.1f}")
    for level, m in summary["per_level"].items():
        print(f"    {level}: success={m['success']}/{m['n']} "
              f"({m['success_rate']:.1%}) steps={m['mean_steps']:.1f} "
              f"coll={m['mean_collisions']:.1f} "
              f"optimality={m['optimality']:.2f}")
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--levels", default="L1",
                    help="逗号分隔，例如 L1 或 L1,L2,L3,L4")
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--maps-per-update", type=int, default=1)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--memory", type=int, default=3)
    ap.add_argument("--visited", action="store_true",
                    help="把已走过的格子作为历史足迹放进观测")
    ap.add_argument("--revisit-penalty", type=float, default=0.0,
                    help="多步重复访问已走格子的额外奖励；0 表示关闭")
    ap.add_argument("--max-group-size", type=int, default=8,
                    help="多步训练单个 forward chunk 的最大样本数")
    ap.add_argument("--obs-style", default="minimal")
    ap.add_argument("--fewshot", default="none",
                    choices=["none", "rel", "rel_l1"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--val-limit", type=int, default=10,
                    help="每级固定 val 图数量；0 表示不评估")
    ap.add_argument("--test-limit", type=int, default=10,
                    help="每级固定 test 图数量；0 表示不评估")
    ap.add_argument("--eval-batch", type=int, default=1,
                    help="评估时一次并行跑多少张地图")
    ap.add_argument("--ui-url", default=None)
    ap.add_argument("--ui-host", default="127.0.0.1",
                    help="自带 UI 时监听地址；远程访问可设 0.0.0.0")
    ap.add_argument("--ui-port", type=int, default=7862)
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--json", default="out/train_grpo.json")
    ap.add_argument("--eval-json", default="out/eval_grpo.json")
    ap.add_argument("--save", default="out/lora_grpo.pt")
    ap.add_argument("--best-save", default="out/lora_grpo_best.pt")
    ap.add_argument("--init-adapter", default=None,
                    help="可选：从 SFT checkpoint 初始化 LoRA")
    args = ap.parse_args()

    try:
        levels = parse_levels(args.levels)
    except ValueError as exc:
        ap.error(str(exc))
    fewshot = None if args.fewshot == "none" else args.fewshot
    if fewshot in ("rel", "rel_l1") and args.obs_style != "rel":
        ap.error("--fewshot rel/rel_l1 需要同时使用 --obs-style rel")

    ds = MapDataset.load(Path(args.data) / "maps" / f"{args.split}.jsonl")
    val_ds = MapDataset.load(Path(args.data) / "maps" / "val.jsonl")
    test_ds = MapDataset.load(Path(args.data) / "maps" / "test_id.jsonl")
    val_maps = pick_eval_maps(val_ds, levels, args.val_limit)
    test_maps = pick_eval_maps(test_ds, levels, args.test_limit)
    rng = random.Random(args.seed)

    viz = None
    if not args.no_ui:
        viz = TrainViz(args.ui_url, host=args.ui_host, port=args.ui_port,
                       serve=args.ui_url is None)
        print(f"UI: {viz.url}")

    print(f"levels={levels}  fewshot={fewshot}  "
          f"max_seq_len={args.max_seq_len}  memory={args.memory}")
    print(f"val_maps={len(val_maps)}  test_maps={len(test_maps)}")
    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    tie_embeddings(model)

    engine = RolloutEngine(model, tokenizer,
                           max_batch=max(args.g, args.eval_batch),
                           max_seq_len=args.max_seq_len, kv_ratio=0.5,
                           device=device)
    train_sampling = Sampling(max_tokens=args.max_tokens,
                              temperature=args.temperature)
    policy = EnginePolicy(engine, tokenizer, train_sampling, fewshot=fewshot)
    eval_policy = EnginePolicy(
        engine, tokenizer,
        Sampling(max_tokens=args.max_tokens, temperature=0.0),
        fewshot=fewshot,
    )
    adapter = LoRAAdapter(model, r=args.lora_r, alpha=2 * args.lora_r)
    trainer = MultiStepGRPOTrainer(
        adapter, TrainConfig(lr=args.lr, clip_eps=args.clip, beta=args.beta))
    if args.init_adapter:
        blob = torch.load(args.init_adapter, map_location=device)
        adapter.load_state_dict(blob.get("adapter", blob))
        print(f"初始化 adapter ← {args.init_adapter}")
    print(f"LoRA {sum(p.numel() for p in adapter.trainable_parameters())/1e6:.2f}M  "
          f"KV 池 {engine.pool_stats}")

    val_history: list[dict] = []
    test_metrics: dict | None = None
    best_val = -1.0
    hist: list[dict] = []

    # update 0 的固定 val 基线
    if val_maps:
        print("\n[val @0]")
        val_metrics, _ = evaluate_maps(
            eval_policy, val_maps, obs_style=args.obs_style,
            memory=args.memory, show_visited=args.visited,
            revisit_penalty=args.revisit_penalty,
            batch_size=args.eval_batch, split="val")
        val_history.append({"update": 0, **val_metrics})
        best_val = val_metrics["success_rate"]
        Path(args.best_save).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.best_save)
        _cuda_cleanup()

    t_start = time.time()
    try:
        for u in range(1, args.updates + 1):
            _cuda_cleanup()
            maps = ds.sample(args.maps_per_update, rng, levels=levels)
            samples = []
            returns, successes, steps_used, collisions, fmterr = [], 0, [], 0, 0
            revisits = 0
            action_counts = Counter()
            degenerate = 0
            for mi, rec in enumerate(maps):
                maze = rec.to_maze()
                trajs = run_group(
                    maze, policy, args.g, ui=viz,
                    episode_prefix=f"u{u}m{mi}", obs_style=args.obs_style,
                    memory=args.memory, show_visited=args.visited,
                    revisit_penalty=args.revisit_penalty,
                )
                group_returns = [t.total_reward for t in trajs]
                advs, degen = group_advantages(group_returns)
                returns.extend(group_returns)
                if degen:
                    degenerate += 1
                else:
                    for traj, adv in zip(trajs, advs):
                        samples.extend(samples_from_trajectory(
                            traj, adv, episode=f"u{u}m{mi}"))
                for t in trajs:
                    successes += int(t.success)
                    steps_used.append(t.steps)
                    for x in t.turns:
                        if x.action in ACTIONS:
                            action_counts[x.action] += 1
                        if x.info.get("revisit"):
                            revisits += 1
                        if x.info.get("reason") == "collision":
                            collisions += 1
                        if x.action == "invalid":
                            fmterr += 1

            stats = None
            if samples:
                stats = trainer.update(
                    samples, max_group_size=args.max_group_size)
                _cuda_cleanup()

            rec_metrics = {
                "update": u,
                "total_updates": args.updates,
                "episodes": len(returns),
                "mean_reward": round(statistics.mean(returns), 2),
                "success": successes,
                "success_rate": round(successes / len(returns), 4)
                                if returns else 0.0,
                "mean_steps": round(statistics.mean(steps_used), 1),
                "collisions": collisions,
                "format_errors": fmterr,
                "revisits": revisits,
                "action_counts": dict(action_counts),
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

            do_eval = (args.eval_every > 0
                       and (u % args.eval_every == 0
                            or u == args.updates))
            if do_eval and val_maps:
                print(f"\n[val @{u}]")
                val_metrics, _ = evaluate_maps(
                    eval_policy, val_maps, obs_style=args.obs_style,
                    memory=args.memory, show_visited=args.visited,
                    revisit_penalty=args.revisit_penalty,
            batch_size=args.eval_batch, split="val")
                val_history.append({"update": u, **val_metrics})
                rec_metrics["val"] = val_metrics
                _cuda_cleanup()
                if val_metrics["success_rate"] > best_val:
                    best_val = val_metrics["success_rate"]
                    Path(args.best_save).parent.mkdir(
                        parents=True, exist_ok=True)
                    trainer.save(args.best_save)
                    rec_metrics["best"] = True

            hist.append(rec_metrics)
            print(f"[{u}/{args.updates}] " + "  ".join(
                f"{k}={v}" for k, v in rec_metrics.items()
                if k != "update" and k != "val"))
            if viz is not None:
                viz.update(**rec_metrics)

        if test_maps:
            print("\n[test final]")
            test_metrics, _ = evaluate_maps(
                eval_policy, test_maps, obs_style=args.obs_style,
                memory=args.memory, show_visited=args.visited,
                revisit_penalty=args.revisit_penalty,
                batch_size=args.eval_batch, split="test")
    finally:
        dt = time.time() - t_start
        print(f"\n总耗时 {dt/60:.1f} min")
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.save)
        out = {
            "config": {
                "levels": list(levels),
                "fewshot": fewshot,
                "obs_style": args.obs_style,
                "updates": args.updates,
                "maps_per_update": args.maps_per_update,
                "g": args.g,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "max_seq_len": args.max_seq_len,
                "memory": args.memory,
                "visited": args.visited,
                "revisit_penalty": args.revisit_penalty,
                "max_group_size": args.max_group_size,
                "seed": args.seed,
                "val_limit": args.val_limit,
                "test_limit": args.test_limit,
                "eval_batch": args.eval_batch,
            },
            "train": hist,
            "val": val_history,
            "test": test_metrics,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(args.eval_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.eval_json).write_text(
            json.dumps({"val": val_history, "test": test_metrics},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"adapter → {args.save}\nbest → {args.best_save}")
        print(f"日志 → {args.json}\n评估 → {args.eval_json}")
        engine.close()
        if viz is not None:
            viz.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
