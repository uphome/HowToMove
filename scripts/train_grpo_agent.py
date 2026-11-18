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
    ap.add_argument("--level", default="L1")
    ap.add_argument("--updates", type=int, default=10)
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

    train_records = [
        r for r in load_records(Path(args.data) / "maps" / "train.jsonl")
        if r.level == args.level
    ]
    train_records = sorted(train_records, key=lambda r: r.hash)
    if not train_records:
        raise RuntimeError(f"no training maps for {args.level}")

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

    history = []
    t0 = time.time()
    for update in range(1, args.updates + 1):
        rec, runs = run_group(
            train_records, rollout_policy,
            g=args.g, history_window=args.history_window,
            index=(update - 1) % len(train_records),
        )
        rewards = [float(run.total_reward) for run in runs]
        advantages, degenerate = group_advantages(rewards)
        success = [bool(run.success) for run in runs]
        row = {
            "update": update,
            "map_id": rec.id,
            "successes": sum(success),
            "rewards": [round(x, 4) for x in rewards],
            "advantage_std": round(
                math.sqrt(sum((r - sum(rewards) / len(rewards)) ** 2
                              for r in rewards) / len(rewards)), 4
            ),
            "degenerate": degenerate,
        }
        if degenerate:
            print(f"[{update}/{args.updates}] {rec.id[:12]} "
                  f"skip degenerate success={sum(success)}/{args.g}")
            history.append(row)
            continue

        samples = []
        for run, adv in zip(runs, advantages):
            samples.extend(samples_from_run(run, adv))
        if not samples:
            print(f"[{update}/{args.updates}] no samples, skip")
            history.append({**row, "skipped": "no_samples"})
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
        print(f"[{update}/{args.updates}] {rec.id[:12]} "
              f"success={sum(success)}/{args.g} "
              f"tokens={stats.tokens} loss={stats.loss:.4f} "
              f"pg={stats.pg:.4f} clip={stats.clip_frac:.3f} "
              f"drop={stats.drop_frac:.3f} grad={stats.grad_norm:.3f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grpo_agent.pt"
    trainer.save(str(save_path))
    print(f"adapter -> {save_path}")

    engine.close()
    engine = RolloutEngine(
        model, tokenizer, max_batch=args.g,
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device,
    )
    eval_policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=0.0),
    )
    val_records = load_records(Path(args.data) / "maps" / "val.jsonl")
    test_records = load_records(Path(args.data) / "maps" / "test.jsonl")
    val = evaluate_agent(
        eval_policy, pick_maps(val_records, (args.level,), args.val_limit),
        history_window=args.history_window, split="val",
        batch_size=args.g,
    )
    test = evaluate_agent(
        eval_policy, pick_maps(test_records, (args.level,), args.test_limit),
        history_window=args.history_window, split="test",
        batch_size=args.g,
    )
    print(f"val: {val}")
    print(f"test: {test}")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "init_adapter": args.init_adapter,
        "level": args.level,
        "updates": args.updates,
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
