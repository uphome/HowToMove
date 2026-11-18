"""Probe group-relative rollout diversity for the tool-calling agent.

For each selected map, sample G trajectories from an SFT checkpoint and
report whether the group contains both successes and failures. GRPO needs
within-group reward variation; this script measures that before training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401
import torch  # noqa: E402

from agent import AgentGroup, ChatPolicy  # noqa: E402
from dataset import load_records  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--level", default="L1")
    ap.add_argument("--maps", type=int, default=20)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="/tmp/opencode/grpo_signal_v3_l1.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, device = load_model(args.model)
    tie_embeddings(model)
    blob = torch.load(args.init_adapter, map_location=device)
    adapter = LoRAAdapter(model, r=16, alpha=32)
    adapter.load_state_dict(blob.get("adapter", blob))
    model.eval()

    records = [
        r for r in load_records(Path(args.data) / "maps" / "train.jsonl")
        if r.level == args.level
    ]
    records = sorted(records, key=lambda r: r.hash)[:args.maps]

    engine = RolloutEngine(
        model, tokenizer, max_batch=args.g,
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device,
    )
    policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=args.temperature,
                 top_p=args.top_p),
    )

    rows = []
    t0 = time.time()
    for index, rec in enumerate(records, 1):
        group = AgentGroup(
            rec.to_maze(), policy, g=args.g,
            history_window=args.history_window,
            episode_prefix=f"grpo-probe/{rec.id}",
        )
        runs = group.run()
        successes = [bool(run.success) for run in runs]
        rewards = [round(run.total_reward, 4) for run in runs]
        collisions = [
            sum(1 for t in run.turns if t["tool_result"]["blocked"])
            for run in runs
        ]
        steps = [run.steps for run in runs]
        format_errors = [
            sum(1 for t in run.turns if t["action"] is None)
            for run in runs
        ]
        success_count = sum(successes)
        mixed = 0 < success_count < args.g
        row = {
            "map_id": rec.id,
            "level": rec.level,
            "shortest": rec.shortest,
            "successes": success_count,
            "mixed": mixed,
            "all_fail": success_count == 0,
            "all_success": success_count == args.g,
            "rewards": rewards,
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "reward_range": round(max(rewards) - min(rewards), 4),
            "steps": steps,
            "collisions": collisions,
            "format_errors": format_errors,
        }
        rows.append(row)
        print(
            f"[{index}/{len(records)}] {rec.id[:12]} "
            f"success={success_count}/{args.g} "
            f"reward={row['mean_reward']:.3f} "
            f"range={row['reward_range']:.3f} "
            f"coll={sum(collisions)}"
        )

    mixed = sum(r["mixed"] for r in rows)
    all_fail = sum(r["all_fail"] for r in rows)
    all_success = sum(r["all_success"] for r in rows)
    summary = {
        "model": args.model,
        "adapter": str(args.init_adapter),
        "level": args.level,
        "maps": len(rows),
        "g": args.g,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "elapsed_sec": round(time.time() - t0, 1),
        "mixed_groups": mixed,
        "all_fail_groups": all_fail,
        "all_success_groups": all_success,
        "mixed_rate": round(mixed / len(rows), 4) if rows else 0.0,
        "mean_reward_range": round(
            sum(r["reward_range"] for r in rows) / len(rows), 4
        ) if rows else 0.0,
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(
        f"summary: mixed={mixed}/{len(rows)} all_fail={all_fail} "
        f"all_success={all_success} "
        f"mean_reward_range={summary['mean_reward_range']}"
    )
    print(f"json -> {out}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
