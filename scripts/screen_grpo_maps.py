"""Greedy difficulty screen for a unified GRPO map pool.

Run the final SFT policy once per train map and record success, collisions,
steps, format errors, and reward. The result is the first-stage pool filter;
hard maps can then be probed with G>1 rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401
import torch  # noqa: E402

from agent import ChatPolicy, MazeAgent  # noqa: E402
from dataset import load_records  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402


def evaluate_chunk(policy, records, *, history_window: int,
                   split: str) -> list[dict]:
    agents = [
        MazeAgent(rec.to_maze(), history_window=history_window,
                  episode_id=f"{split}/{rec.id}")
        for rec in records
    ]
    active = list(agents)
    while active:
        steps = policy.act_batch([a.messages() for a in active])
        for agent, step in zip(active, steps):
            agent.apply(step)
        active = [a for a in active if not a.done]

    rows = []
    for rec, agent in zip(records, agents):
        run = agent.to_run()
        collisions = sum(1 for t in run.turns
                         if t["tool_result"]["blocked"])
        actions = Counter(t["action"] for t in run.turns
                          if t["action"] is not None)
        rows.append({
            "map_id": rec.id,
            "level": rec.level,
            "shortest": rec.shortest,
            "detour": rec.detour,
            "walls": len(rec.walls),
            "success": bool(run.success),
            "truncated": bool(run.truncated),
            "steps": run.steps,
            "reward": round(run.total_reward, 4),
            "collisions": collisions,
            "format_errors": sum(1 for t in run.turns
                                  if t["action"] is None),
            "actions": dict(actions),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--init-adapter", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--levels", default="L1,L2,L3,L4")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 means all maps")
    ap.add_argument("--out", default="/tmp/opencode/grpo_greedy_screen.json")
    ap.add_argument("--rows-jsonl", default=None,
                    help="incremental per-map results; defaults next to --out")
    ap.add_argument("--resume", action="store_true",
                    help="skip map ids already present in --rows-jsonl")
    args = ap.parse_args()

    levels = tuple(x.strip().upper() for x in args.levels.split(",")
                   if x.strip())
    model, tokenizer, device = load_model(args.model)
    tie_embeddings(model)
    blob = torch.load(args.init_adapter, map_location=device)
    adapter = LoRAAdapter(model, r=16, alpha=32)
    adapter.load_state_dict(blob.get("adapter", blob))
    model.eval()

    records = [r for r in load_records(
        Path(args.data) / "maps" / f"{args.split}.jsonl")
        if r.level in levels]
    records = sorted(records, key=lambda r: r.hash)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise RuntimeError("no records selected")

    rows_path = (Path(args.rows_jsonl) if args.rows_jsonl
                 else Path(args.out).with_suffix(".rows.jsonl"))
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    prior_rows = []
    done_ids = set()
    if args.resume and rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prior_rows.append(row)
                done_ids.add(row["map_id"])
        print(f"resume: {len(done_ids)} rows from {rows_path}")
    else:
        rows_path.write_text("", encoding="utf-8")

    engine = RolloutEngine(
        model, tokenizer, max_batch=max(1, args.batch_size),
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device,
    )
    policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=0.0),
    )

    rows = list(prior_rows)
    pending = [r for r in records if r.id not in done_ids]
    t0 = time.time()
    with rows_path.open("a", encoding="utf-8", newline="\n") as fh:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            new_rows = evaluate_chunk(
                policy, chunk, history_window=args.history_window,
                split=args.split,
            )
            for row in new_rows:
                fh.write(json.dumps(
                    row, ensure_ascii=False, sort_keys=True,
                ) + "\n")
                rows.append(row)
            fh.flush()
            done = min(start + len(chunk), len(pending))
            print(f"[{len(rows)}/{len(records)}] "
                  f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    by_level = {}
    for level in levels:
        sub = [r for r in rows if r["level"] == level]
        by_level[level] = {
            "n": len(sub),
            "easy": sum(1 for r in sub if r["success"]),
            "hard": sum(1 for r in sub if not r["success"]),
            "format_errors": sum(r["format_errors"] for r in sub),
            "mean_collisions": round(
                sum(r["collisions"] for r in sub) / len(sub), 3
            ) if sub else 0.0,
            "mean_steps": round(
                sum(r["steps"] for r in sub) / len(sub), 3
            ) if sub else 0.0,
        }
    summary = {
        "checkpoint": str(args.init_adapter),
        "split": args.split,
        "levels": list(levels),
        "rows_jsonl": str(rows_path),
        "n": len(rows),
        "elapsed_sec": round(time.time() - t0, 1),
        "overall": {
            "easy": sum(1 for r in rows if r["success"]),
            "hard": sum(1 for r in rows if not r["success"]),
            "format_errors": sum(r["format_errors"] for r in rows),
        },
        "by_level": by_level,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"summary: {json.dumps(summary['overall'], ensure_ascii=False)}")
    for level, m in by_level.items():
        print(f"  {level}: {m}")
    print(f"json -> {out}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
