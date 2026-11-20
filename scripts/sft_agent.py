"""SFT for the static-map tool-calling agent.

Maps are loaded from data/maps/{train,val,test}.jsonl. Trajectories are
generated from a map on the fly; no trajectory dataset is stored.

    python scripts/sft_agent.py --levels L0 --maps 100 --epochs 1

Use ``--eval-only --init-adapter ...`` to evaluate an existing adapter
without rebuilding or training again. ``--plot-dir`` saves the loss curve.
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

from agent import ChatPolicy, MazeAgent  # noqa: E402
from agent.sft_data import build_agent_sft_items  # noqa: E402
from dataset import load_records  # noqa: E402
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
                   split: str, batch_size: int = 1) -> dict:
    """Evaluate task records, batching independent agents when possible."""
    rows = []
    batch_size = max(1, batch_size)
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        agents = [
            MazeAgent(
                rec.to_maze(),
                history_window=history_window,
                episode_id=f"{split}-{start + j + 1}",
            )
            for j, rec in enumerate(chunk)
        ]
        active = list(agents)
        while active:
            messages = [agent.messages() for agent in active]
            batch_fn = getattr(policy, "act_batch", None)
            if batch_fn is None:
                steps = [policy.act(m) for m in messages]
            else:
                steps = batch_fn(messages)
            if len(steps) != len(active):
                raise RuntimeError(
                    f"policy returned {len(steps)} steps for "
                    f"{len(active)} active agents")
            for agent, step in zip(active, steps):
                agent.apply(step)
            active = [agent for agent in active if not agent.done]

        for j, (rec, agent) in enumerate(zip(chunk, agents), start + 1):
            run = agent.to_run()
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
            print(f"  [{split} {j}/{len(records)}] {rec.id:16s} "
                  f"steps={r['steps']:2d} success={str(r['success']):5s} "
                  f"coll={r['collisions']:2d} fmt={r['format_errors']} "
                  f"actions={dict(actions)}")
    def aggregate(selected):
        n = len(selected)
        ok = sum(r["success"] for r in selected)
        actions = Counter()
        for r in selected:
            actions.update(r["actions"])
        return {
            "n": n,
            "success": ok,
            "success_rate": round(ok / n, 4) if n else 0.0,
            "mean_steps": round(
                statistics.mean(r["steps"] for r in selected), 1
            ) if selected else 0.0,
            "collisions": sum(r["collisions"] for r in selected),
            "format_errors": sum(r["format_errors"] for r in selected),
            "actions": dict(actions),
        }

    result = aggregate(rows)
    result["by_level"] = {
        level: aggregate([r for r in rows if r["level"] == level])
        for level in sorted({r["level"] for r in rows})
    }
    return result


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


def moving_average(values: list[float], window: int) -> list[float]:
    """Centered moving average for a noisy per-step training loss."""
    window = max(1, int(window))
    n = len(values)
    if window == 1 or n < 2:
        return list(values)
    half = window // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def plot_loss(history: list[dict], out_dir: str, *,
              smooth_window: int = 50,
              filename: str = "sft_loss.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = [x["step"] for x in history]
    losses = [x["loss"] for x in history]
    smooth = moving_average(losses, smooth_window)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if smooth_window > 1:
        ax.plot(steps, losses, color="#9ecae1", linewidth=0.7,
                alpha=0.55, label="per-step loss")
    ax.plot(steps, smooth, color="#1f77b4", linewidth=1.9,
            label=f"moving average (w={smooth_window})")
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy loss")
    ax.set_title("Agent SFT loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)


def train_sft(items, model, tokenizer, device, *, epochs: int,
              batch_size: int, lr: float, seed: int,
              init_state: dict | None = None
              ) -> tuple[LoRAAdapter, int, list[dict]]:
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
    loss_history = []
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
                value = float(loss.detach())
                losses.append(value)
                loss_history.append({
                    "epoch": epoch,
                    "step": step,
                    "loss": value,
                })
                if step % 20 == 0:
                    print(f"epoch={epoch} step={step} "
                          f"loss={statistics.mean(losses[-20:]):.4f}")
        print(f"epoch {epoch} done loss={statistics.mean(losses):.4f} "
              f"elapsed={(time.time()-t0)/60:.1f} min")
    model.model.grad_checkpoint = False
    model.eval()
    return adapter, step, loss_history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--levels", default="L0")
    ap.add_argument("--maps", type=int, default=800,
                    help="每级使用的 map 数")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--recovery-per-task", type=int, default=0,
                    help="每个 task 额外注入的撞墙恢复样本数")
    ap.add_argument("--recovery-depth", type=int, default=1,
                    help="每个恢复状态沿最优路径继续训练的步数")
    ap.add_argument("--val-limit", type=int, default=10)
    ap.add_argument("--test-limit", type=int, default=10)
    ap.add_argument("--eval-batch-size", type=int, default=1,
                    help="并行评估的独立 agent 数")
    ap.add_argument("--eval-only", action="store_true",
                    help="只加载 --init-adapter 并评估，不训练")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--plot-dir", default=None,
                    help="可选：保存 sft_loss.png 的目录")
    ap.add_argument("--smooth-window", type=int, default=50,
                    help="loss 曲线移动平均窗口")
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

    if args.eval_only:
        if args.init_adapter:
            blob = torch.load(args.init_adapter, map_location=device)
            adapter = LoRAAdapter(model, r=16, alpha=32)
            adapter.load_state_dict(blob.get("adapter", blob))
            save_path = Path(args.init_adapter)
            print(f"评估 adapter ← {save_path}")
        else:
            adapter = None
            save_path = None
            print("评估 base model（无 adapter）")
        model.eval()
        items: list = []
        actions: Counter = Counter()
        n_maps = 0
        loss_history: list[dict] = []
    else:
        train_tasks = load_records(
            Path(args.data) / "maps" / "train.jsonl")
        items, actions, n_maps = build_agent_sft_items(
            train_tasks, tokenizer,
            levels=levels,
            maps_per_level=args.maps,
            history_window=args.history_window,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            recovery_per_task=args.recovery_per_task,
            recovery_depth=args.recovery_depth,
        )
        print(f"maps={n_maps} sft_items={len(items)} "
              f"actions={dict(actions)}")
        if not items:
            raise RuntimeError("no SFT items generated")

        init_state = None
        if args.init_adapter:
            blob = torch.load(args.init_adapter, map_location=device)
            init_state = blob.get("adapter", blob)
            print(f"初始化 adapter ← {args.init_adapter}")
        adapter, steps, loss_history = train_sft(
            items, model, tokenizer, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed, init_state=init_state,
        )
        if args.plot_dir:
            plot_loss(loss_history, args.plot_dir,
                      smooth_window=args.smooth_window)
            print(f"loss plot → {Path(args.plot_dir) / 'sft_loss.png'}")

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
                "recovery_per_task": args.recovery_per_task,
                "recovery_depth": args.recovery_depth,
                "smooth_window": args.smooth_window,
                "seed": args.seed,
            },
        }, save_path)
        print(f"adapter → {save_path}")

    # Evaluate the exact agent/tool protocol, greedy.
    eval_batch = max(1, args.eval_batch_size)
    engine = RolloutEngine(
        model, tokenizer, max_batch=eval_batch,
        max_seq_len=args.max_seq_len, kv_ratio=0.5, device=device)
    policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=0.0),
    )
    val_tasks = load_records(Path(args.data) / "maps" / "val.jsonl")
    test_tasks = load_records(Path(args.data) / "maps" / "test.jsonl")
    val = evaluate_agent(
        policy, pick_maps(val_tasks, levels, args.val_limit),
        history_window=args.history_window, split="val",
        batch_size=eval_batch)
    test = evaluate_agent(
        policy, pick_maps(test_tasks, levels, args.test_limit),
        history_window=args.history_window, split="test",
        batch_size=eval_batch)
    print(f"val: {val}")
    print(f"test: {test}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({
        "maps": n_maps,
        "items": len(items),
        "train_actions": dict(actions),
        "loss_history": loss_history,
        "val": val,
        "test": test,
        "eval_batch_size": eval_batch,
        "adapter": str(save_path) if save_path is not None else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
