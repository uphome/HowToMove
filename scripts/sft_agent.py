"""SFT for the static-map tool-calling agent.

Trajectories are generated from data/maps/*.jsonl on the fly; no state or
trajectory dataset is stored under data/.

    python scripts/sft_agent.py --levels L0 --maps 100 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

import torch  # noqa: E402

from agent import AgentGroup, ChatPolicy, load_tasks  # noqa: E402
from agent.sft_data import build_agent_sft_items  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import LEVELS  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.forward import PAD_ID, completion_mask, token_logprobs  # noqa: E402


def pick_maps(records, levels, per_level: int):
    out = []
    for level in levels:
        pool = sorted((r for r in records if r.level == level),
                      key=lambda r: r.hash)
        out.extend(pool[:per_level])
    return out


def evaluate_agent(policy, records, *, history_window: int,
                   split: str) -> dict:
    rows = []
    for i, rec in enumerate(records, 1):
        group = AgentGroup(
            rec.to_maze(), policy, g=1,
            history_window=history_window,
            episode_prefix=f"{split}-{i}",
        )
        run = group.run()[0]
        actions = Counter(t["action"] for t in run.turns
                          if t["action"] is not None)
        rows.append({
            "id": rec.id,
            "level": rec.level,
            "shortest": rec.shortest,
            "steps": run.steps,
            "success": bool(run.success),
            "truncated": bool(run.truncated),
            "reward": round(run.total_reward, 4),
            "collisions": sum(1 for t in run.turns
                              if t["tool_result"]["blocked"]),
            "format_errors": sum(1 for t in run.turns
                                 if t["action"] is None),
            "actions": dict(actions),
        })
        r = rows[-1]
        print(f"  [{split} {i}/{len(records)}] {rec.id:16s} "
              f"steps={r['steps']:2d} success={str(r['success']):5s} "
              f"coll={r['collisions']:2d} fmt={r['format_errors']} "
              f"actions={dict(actions)}")
    n = len(rows)
    ok = sum(r["success"] for r in rows)
    all_actions = Counter()
    for r in rows:
        all_actions.update(r["actions"])
    return {
        "n": n,
        "success": ok,
        "success_rate": round(ok / n, 4) if n else 0.0,
        "mean_steps": round(statistics.mean(r["steps"] for r in rows), 1)
                       if rows else 0.0,
        "collisions": sum(r["collisions"] for r in rows),
        "format_errors": sum(r["format_errors"] for r in rows),
        "actions": dict(all_actions),
    }


def _pad_batch(chunk, device):
    p = len(chunk[0].prompt_ids)
    t = max(len(x.completion_ids) for x in chunk)
    ids = torch.full((len(chunk), p + t), PAD_ID,
                     dtype=torch.long, device=device)
    lens = []
    for i, item in enumerate(chunk):
        if len(item.prompt_ids) != p:
            raise RuntimeError("prompt length bucket mismatch")
        n = len(item.completion_ids)
        lens.append(n)
        ids[i, :p] = torch.tensor(item.prompt_ids, dtype=torch.long,
                                  device=device)
        ids[i, p:p + n] = torch.tensor(item.completion_ids, dtype=torch.long,
                                       device=device)
    return ids, p, lens


def train_sft(items, model, tokenizer, device, *, epochs: int,
              batch_size: int, lr: float, seed: int,
              init_state: dict | None = None) -> tuple[LoRAAdapter, int]:
    rng = random.Random(seed)
    adapter = LoRAAdapter(model, r=16, alpha=32)
    if init_state is not None:
        adapter.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(adapter.param_groups(lr))
    model.model.grad_checkpoint = True
    model.train()

    buckets = defaultdict(list)
    for item in items:
        buckets[len(item.prompt_ids)].append(item)
    print(f"SFT prompt buckets={len(buckets)} "
          f"range={min(buckets)}..{max(buckets)}")

    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        losses = []
        for plen in sorted(buckets):
            bucket = buckets[plen]
            rng.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                chunk = bucket[i:i + batch_size]
                ids, p, lens = _pad_batch(chunk, device)
                lp = token_logprobs(model, ids, p, lens)
                mask = completion_mask(lens, device=device)
                loss = -(lp * mask).sum() / mask.sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    adapter.trainable_parameters(), 1.0)
                optimizer.step()
                step += 1
                losses.append(float(loss.detach()))
                if step % 20 == 0:
                    print(f"epoch={epoch} step={step} "
                          f"loss={statistics.mean(losses[-20:]):.4f}")
        print(f"epoch {epoch} done loss={statistics.mean(losses):.4f} "
              f"elapsed={(time.time()-t0)/60:.1f} min")
    model.model.grad_checkpoint = False
    model.eval()
    return adapter, step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--levels", default="L0")
    ap.add_argument("--maps", type=int, default=100,
                    help="每级使用的地图数")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--val-limit", type=int, default=10)
    ap.add_argument("--test-limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--save-dir", default="/tmp/opencode/sft_agent")
    ap.add_argument("--json", default="/tmp/opencode/sft_agent.json")
    args = ap.parse_args()

    levels = tuple(x.strip().upper() for x in args.levels.split(",")
                   if x.strip())
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        ap.error(f"unknown levels: {bad}")

    print(f"levels={levels} maps_per_level={args.maps} "
          f"history_window={args.history_window}")
    model, tokenizer, device = load_model(args.model)
    tie_embeddings(model)

    train_tasks = load_tasks(args.data, "train")
    items, actions, n_maps = build_agent_sft_items(
        train_tasks, tokenizer,
        levels=levels,
        maps_per_level=args.maps,
        history_window=args.history_window,
        max_seq_len=args.max_seq_len,
    )
    print(f"maps={n_maps} sft_items={len(items)} actions={dict(actions)}")
    if not items:
        raise RuntimeError("no SFT items generated")

    init_state = None
    if args.init_adapter:
        blob = torch.load(args.init_adapter, map_location=device)
        init_state = blob.get("adapter", blob)
        print(f"初始化 adapter ← {args.init_adapter}")
    adapter, steps = train_sft(
        items, model, tokenizer, device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, init_state=init_state,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "sft_agent.pt"
    torch.save({
        "adapter": adapter.state_dict(),
        "updates": steps,
        "config": {
            "levels": list(levels),
            "maps_per_level": args.maps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "history_window": args.history_window,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
        },
    }, save_path)
    print(f"adapter → {save_path}")

    # Evaluate the exact agent/tool protocol, greedy.
    engine = RolloutEngine(
        model, tokenizer, max_batch=1,
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device)
    policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=0.0),
    )
    val_tasks = load_tasks(args.data, "val")
    test_tasks = load_tasks(args.data, "test")
    val = evaluate_agent(
        policy, pick_maps(val_tasks, levels, args.val_limit),
        history_window=args.history_window, split="val")
    test = evaluate_agent(
        policy, pick_maps(test_tasks, levels, args.test_limit),
        history_window=args.history_window, split="test")
    print(f"val: {val}")
    print(f"test: {test}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({
        "maps": n_maps,
        "items": len(items),
        "train_actions": dict(actions),
        "val": val,
        "test": test,
        "adapter": str(save_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
