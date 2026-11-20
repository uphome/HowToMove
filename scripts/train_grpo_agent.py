"""Minimal Agent GRPO for the static-map tool-calling maze task.

Rollout protocol is the same as SFT/evaluation:

    system: static map
    user:   Begin the maze.
    assistant: move(...)
    tool: compact JSON

For each training map, sample G complete trajectories, compute group-relative
advantages from total reward, flatten every turn into a GRPO Sample, and update
the same LoRA adapter used by the rollout policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paths  # noqa: E402,F401
import torch  # noqa: E402

from agent import AgentGroup, ChatPolicy  # noqa: E402
from dataset import load_records  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.multistep import MultiStepGRPOTrainer  # noqa: E402
from grpo.train import Sample, TrainConfig, group_advantages  # noqa: E402
from sft_agent import evaluate_agent, pick_maps  # noqa: E402


def pool_sequence(rows: list[dict]) -> list[dict]:
    """Round-robin levels so the unified pool is not L4-dominated."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["level"], []).append(row)
    for values in groups.values():
        values.sort(key=lambda r: r["map_id"])
    levels = sorted(groups)
    out = []
    while True:
        added = False
        for level in levels:
            if groups[level]:
                out.append(groups[level].pop(0))
                added = True
        if not added:
            return out


def metric_points(history: list[dict], key: str) -> tuple[list[int],
                                                          list[float]]:
    xs, ys = [], []
    for row in history:
        value = row.get(key)
        if value is None:
            continue
        xs.append(int(row["update"]))
        ys.append(float(value))
    return xs, ys


def plot_metrics(history: list[dict], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def one(key: str, title: str, filename: str) -> None:
        xs, ys = metric_points(history, key)
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, ys, color="#2b6cb0", linewidth=1.8)
        ax.set_xlabel("update")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)

    one("mean_reward", "GRPO mean trajectory reward", "grpo_reward.png")
    one("kl", "GRPO KL", "grpo_kl.png")
    one("loss", "GRPO loss", "grpo_loss.png")
    one("success_rate", "GRPO success rate", "grpo_success.png")

    # Reward curve with min/max band.
    xs = [int(r["update"]) for r in history if "reward_min" in r]
    means = [float(r["mean_reward"]) for r in history
             if "reward_min" in r]
    lows = [float(r["reward_min"]) for r in history
            if "reward_min" in r]
    highs = [float(r["reward_max"]) for r in history
             if "reward_min" in r]
    if xs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, means, color="#2b6cb0", linewidth=1.8,
                label="mean reward")
        ax.fill_between(xs, lows, highs, color="#2b6cb0", alpha=0.15,
                        label="min/max")
        ax.set_xlabel("update")
        ax.set_ylabel("trajectory reward")
        ax.set_title("GRPO reward (group range)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "grpo_reward_range.png", dpi=150)
        plt.close(fig)


def save_metrics(path: Path, history: list[dict], *, pool: str,
                 init_adapter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pool": str(pool),
        "init_adapter": str(init_adapter),
        "history": history,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_group(records, policy, *, g: int, history_window: int,
              index: int):
    rec = records[index]
    group = AgentGroup(
        rec.to_maze(), policy, g=g,
        history_window=history_window,
        revisit_penalty=0.0,
        episode_prefix=f"grpo-train/{rec.id}",
    )
    runs = group.run()
    return rec, runs


def samples_from_run(run, advantage: float) -> list[Sample]:
    samples = []
    for turn in run.turns:
        prompt_ids = turn.get("prompt_ids")
        completion = turn.get("completion_tokens")
        logps = turn.get("token_logprobs")
        if not prompt_ids or not completion or not logps:
            continue
        if len(completion) != len(logps):
            continue
        samples.append(Sample(
            prompt_ids=list(prompt_ids),
            completion_ids=list(completion),
            old_logps=[float(x) for x in logps],
            advantage=float(advantage),
            meta={
                "episode": run.episode_id,
                "turn": turn.get("turn", 0),
            },
        ))
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--pool", default="data/grpo/l1_l4_pool.json",
                    help="unified GRPO pool JSON")
    ap.add_argument("--offset", type=int, default=0,
                    help="start index in the round-robin pool")
    ap.add_argument("--level", default="L1",
                    help="legacy fallback when pool is not used")
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--start-update", type=int, default=0,
                    help="global update number; 0 means after existing metrics")
    ap.add_argument("--resume-metrics", action="store_true",
                    help="append to an existing --json history")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip val/test evaluation (run it separately)")
    ap.add_argument("--eval-levels", default="L1,L2,L3,L4")
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--max-group-size", type=int, default=8)
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--test-limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-dir", default="/tmp/opencode/grpo_agent_l1")
    ap.add_argument("--json", default="/tmp/opencode/grpo_agent_l1.json")
    args = ap.parse_args()

    if args.updates < 1:
        ap.error("updates must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer, device = load_model(args.model)
    tie_embeddings(model)
    blob = torch.load(args.init_adapter, map_location=device)
    adapter = LoRAAdapter(model, r=16, alpha=32)
    adapter.load_state_dict(blob.get("adapter", blob))
    model.eval()

    pool_path = Path(args.pool)
    if not pool_path.exists():
        raise RuntimeError(f"missing GRPO pool: {pool_path}")
    pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
    pool_rows = pool_sequence(pool_payload["rows"])
    if args.offset < 0:
        ap.error("offset must be >= 0")
    selected_rows = pool_rows[args.offset:args.offset + args.updates]
    if not selected_rows:
        raise RuntimeError(
            f"pool exhausted at offset={args.offset}: "
            f"pool size={len(pool_rows)}")

    all_train = load_records(Path(args.data) / "maps" / "train.jsonl")
    by_id = {r.id: r for r in all_train}
    train_records = [by_id[row["map_id"]] for row in selected_rows]
    if len(train_records) != len(selected_rows):
        raise RuntimeError("pool references missing train maps")
    updates = len(train_records)

    trainer = MultiStepGRPOTrainer(
        adapter,
        TrainConfig(lr=args.lr, clip_eps=0.2, beta=0.0,
                    max_grad_norm=1.0, grad_checkpoint=True),
        device=device,
    )

    engine = RolloutEngine(
        model, tokenizer, max_batch=args.g,
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device,
    )
    rollout_policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=args.temperature,
                 top_p=args.top_p),
    )

    metrics_path = Path(args.json)
    history = []
    if args.resume_metrics and metrics_path.exists():
        prior = json.loads(metrics_path.read_text(encoding="utf-8"))
        history = prior.get("history", [])
        print(f"resume metrics: {len(history)} updates from {metrics_path}")
    next_update = (args.start_update
                   or (max((r["update"] for r in history), default=0) + 1))
    t0 = time.time()
    for local_update, rec in enumerate(train_records, 1):
        update = next_update + local_update - 1
        rec, runs = run_group(
            train_records, rollout_policy,
            g=args.g, history_window=args.history_window,
            index=local_update - 1,
        )
        rewards = [float(run.total_reward) for run in runs]
        advantages, degenerate = group_advantages(rewards)
        success = [bool(run.success) for run in runs]
        row = {
            "update": update,
            "map_id": rec.id,
            "level": rec.level,
            "successes": sum(success),
            "success_rate": round(sum(success) / args.g, 4),
            "rewards": [round(x, 4) for x in rewards],
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "reward_min": round(min(rewards), 4),
            "reward_max": round(max(rewards), 4),
            "reward_range": round(max(rewards) - min(rewards), 4),
            "advantage_std": round(
                math.sqrt(sum((r - sum(rewards) / len(rewards)) ** 2
                              for r in rewards) / len(rewards)), 4
            ),
            "degenerate": degenerate,
        }
        if degenerate:
            print(f"[{update}/{next_update + updates - 1}] "
                  f"{rec.id[:12]} skip degenerate "
                  f"success={sum(success)}/{args.g}")
            history.append(row)
            save_metrics(metrics_path, history, pool=pool_path,
                         init_adapter=args.init_adapter)
            continue

        samples = []
        for run, adv in zip(runs, advantages):
            samples.extend(samples_from_run(run, adv))
        if not samples:
            print(f"[{update}/{next_update + updates - 1}] no samples, skip")
            history.append({**row, "skipped": "no_samples"})
            save_metrics(metrics_path, history, pool=pool_path,
                         init_adapter=args.init_adapter)
            continue

        stats = trainer.update(samples, max_group_size=args.max_group_size)
        row.update({
            "samples": len(samples),
            "tokens": stats.tokens,
            "loss": stats.loss,
            "pg": stats.pg,
            "kl": stats.kl,
            "ratio_mean": stats.ratio_mean,
            "clip_frac": stats.clip_frac,
            "drop_frac": stats.drop_frac,
            "grad_norm": stats.grad_norm,
        })
        history.append(row)
        save_metrics(metrics_path, history, pool=pool_path,
                     init_adapter=args.init_adapter)
        print(f"[{update}/{next_update + updates - 1}] {rec.id[:12]} "
              f"success={sum(success)}/{args.g} "
              f"mean_reward={row['mean_reward']:.2f} "
              f"tokens={stats.tokens} loss={stats.loss:.4f} "
              f"kl={stats.kl:.4f} clip={stats.clip_frac:.3f} "
              f"drop={stats.drop_frac:.3f} grad={stats.grad_norm:.3f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grpo_agent.pt"
    trainer.save(str(save_path))
    print(f"adapter -> {save_path}")
    plot_metrics(history, str(save_dir / "curves"))
    save_metrics(metrics_path, history, pool=pool_path,
                 init_adapter=args.init_adapter)

    if args.no_eval:
        engine.close()
        return 0

    engine.close()
    engine = RolloutEngine(
        model, tokenizer, max_batch=args.g,
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device,
    )
    eval_policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=0.0),
    )
    eval_levels = tuple(x.strip().upper() for x in args.eval_levels.split(",")
                        if x.strip())
    val_records = load_records(Path(args.data) / "maps" / "val.jsonl")
    test_records = load_records(Path(args.data) / "maps" / "test.jsonl")
    val = evaluate_agent(
        eval_policy, pick_maps(val_records, eval_levels, args.val_limit),
        history_window=args.history_window, split="val",
        batch_size=args.g,
    )
    test = evaluate_agent(
        eval_policy, pick_maps(test_records, eval_levels, args.test_limit),
        history_window=args.history_window, split="test",
        batch_size=args.g,
    )
    print(f"val: {val}")
    print(f"test: {test}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "init_adapter": args.init_adapter,
        "pool": str(pool_path),
        "pool_offset": args.offset,
        "updates": updates,
        "g": args.g,
        "temperature": args.temperature,
        "lr": args.lr,
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "history": history,
        "val": val,
        "test": test,
        "adapter": str(save_path),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
