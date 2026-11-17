"""多步 SFT 冷启动：用 BFS 最短路生成动作监督。

只做行为克隆：prompt 来自与 rollout 完全相同的观测函数，target 是
最短路上的下一步动作。适合先教模型四方向（尤其 up/down）和多步执行，
再交给 GRPO 做策略优化。

    python scripts/sft_multistep.py \
        --levels L0 --maps 100 --epochs 1 \
        --obs-style rel --fewshot rel --visited

产物是一个 LoRA checkpoint，可直接给 train_grpo.py 的 --init-adapter 使用。
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

import torch  # noqa: E402

from dataset import MapDataset  # noqa: E402
from engine_policy import EnginePolicy  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import ACTIONS, LEVELS  # noqa: E402
from env.render import render_observation  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.forward import PAD_ID, completion_mask, token_logprobs  # noqa: E402
from prompts import render_prompt  # noqa: E402
from rollout import format_tool_call, run_episode  # noqa: E402


@dataclass
class SFTItem:
    prompt_ids: list[int]
    completion_ids: list[int]
    action: str
    map_id: str


def direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    d = (b[0] - a[0], b[1] - a[1])
    for name, delta in ACTIONS.items():
        if delta == d:
            return name
    raise ValueError(f"non-adjacent path step {a} -> {b}")


def build_items(records, tokenizer, *, obs_style: str, fewshot: str | None,
                memory: int, max_maps: int) -> tuple[list[SFTItem], Counter]:
    items: list[SFTItem] = []
    actions = Counter()
    eos = tokenizer.eos_token_id
    for rec in records[:max_maps]:
        maze = rec.to_maze()
        path = maze.shortest_path()
        last: list[str] = []
        visited: list[tuple[int, int]] = []
        for i in range(len(path) - 1):
            pos = path[i]
            nxt = path[i + 1]
            action = direction_between(pos, nxt)
            state = render_observation(
                maze, pos, last, style=obs_style, visited=visited)
            _, prompt_ids = render_prompt(
                tokenizer, state, fewshot=fewshot)
            completion = tokenizer.encode(
                format_tool_call(action), add_special_tokens=False)
            if eos is not None and (not completion or completion[-1] != eos):
                completion.append(eos)
            items.append(SFTItem(
                prompt_ids=prompt_ids,
                completion_ids=completion,
                action=action,
                map_id=rec.id,
            ))
            actions[action] += 1
            last.append(action)
            if len(last) > memory:
                del last[:-memory]
            visited.append(pos)
    return items, actions


def _pad_chunk(chunk: list[SFTItem], device):
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


def evaluate(policy: EnginePolicy, records, *, obs_style: str,
             memory: int, show_visited: bool, split: str) -> dict:
    rows = []
    for i, rec in enumerate(records, 1):
        maze = rec.to_maze()
        traj = run_episode(
            maze, policy, episode=f"sft-{split}-{i}",
            obs_style=obs_style, memory=memory,
            show_visited=show_visited, revisit_penalty=0.0,
        )
        actions = Counter(t.action for t in traj.turns if t.action in ACTIONS)
        rows.append({
            "id": rec.id,
            "success": bool(traj.success),
            "steps": traj.steps,
            "collisions": sum(1 for t in traj.turns
                              if t.info.get("reason") == "collision"),
            "actions": dict(actions),
        })
        print(f"  [{split} {i}/{len(records)}] {rec.id:16s} "
              f"steps={traj.steps:2d} success={str(traj.success):5s} "
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
        "actions": dict(all_actions),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--data", default="data")
    ap.add_argument("--levels", default="L0")
    ap.add_argument("--maps", type=int, default=100,
                    help="每个级别使用的训练地图数")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--obs-style", default="rel")
    ap.add_argument("--fewshot", default="rel",
                    choices=["none", "rel", "rel_l1"])
    ap.add_argument("--memory", type=int, default=3)
    ap.add_argument("--visited", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=1152)
    ap.add_argument("--val-limit", type=int, default=10)
    ap.add_argument("--test-limit", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init-adapter", default=None,
                    help="可选：从已有 LoRA/SFT checkpoint 初始化")
    ap.add_argument("--save-dir", default="/tmp/opencode/m2_sft")
    ap.add_argument("--json", default="/tmp/opencode/m2_sft.json")
    args = ap.parse_args()

    levels = tuple(x.strip().upper() for x in args.levels.split(",")
                   if x.strip())
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        ap.error(f"unknown levels: {bad}")
    fewshot = None if args.fewshot == "none" else args.fewshot
    if fewshot and args.obs_style != "rel":
        ap.error("--fewshot requires --obs-style rel")

    rng = random.Random(args.seed)
    ds = MapDataset.load(Path(args.data) / "maps" / "train.jsonl")
    records = [r for r in ds.records if r.level in levels]
    records = sorted(records, key=lambda r: r.hash)[:args.maps * len(levels)]
    print(f"levels={levels} maps={len(records)} "
          f"obs={args.obs_style} fewshot={fewshot}")

    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    tie_embeddings(model)
    items, action_counts = build_items(
        records, tokenizer, obs_style=args.obs_style, fewshot=fewshot,
        memory=args.memory, max_maps=len(records))
    print(f"SFT samples={len(items)} actions={dict(action_counts)}")
    if not items:
        raise RuntimeError("no SFT samples")

    adapter = LoRAAdapter(model, r=args.lora_r, alpha=2 * args.lora_r)
    if args.init_adapter:
        blob = torch.load(args.init_adapter, map_location=device)
        adapter.load_state_dict(blob.get("adapter", blob))
        print(f"初始化 adapter ← {args.init_adapter}")
    optimizer = torch.optim.AdamW(adapter.param_groups(args.lr))
    model.model.grad_checkpoint = True
    model.train()

    # 按 prompt 长度分桶，保证一个 batch 内 prompt 长度一致。
    buckets: dict[int, list[SFTItem]] = defaultdict(list)
    for item in items:
        buckets[len(item.prompt_ids)].append(item)
    print(f"prompt length buckets={len(buckets)} "
          f"range={min(buckets)}..{max(buckets)}")

    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = []
        for plen in sorted(buckets):
            bucket = buckets[plen]
            rng.shuffle(bucket)
            for i in range(0, len(bucket), args.batch_size):
                chunk = bucket[i:i + args.batch_size]
                ids, p, lens = _pad_chunk(chunk, device)
                lp = token_logprobs(model, ids, p, lens)
                mask = completion_mask(lens, device=device)
                loss = -(lp * mask).sum() / mask.sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    adapter.trainable_parameters(), 1.0)
                optimizer.step()
                step += 1
                epoch_loss.append(float(loss.detach()))
                if step % 20 == 0:
                    print(f"epoch={epoch} step={step} "
                          f"loss={statistics.mean(epoch_loss[-20:]):.4f}")
        print(f"epoch {epoch} done  "
              f"loss={statistics.mean(epoch_loss):.4f}  "
              f"elapsed={(time.time()-t0)/60:.1f} min")

    model.model.grad_checkpoint = False
    model.eval()
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "sft.pt"
    torch.save({
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "updates": step,
        "config": {
            "levels": list(levels),
            "maps": len(records),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "obs_style": args.obs_style,
            "fewshot": fewshot,
            "memory": args.memory,
            "visited": args.visited,
            "init_adapter": args.init_adapter,
        },
    }, save_path)
    print(f"adapter → {save_path}")

    # 固定 val/test 地图评估 SFT 后的动作分布。
    eval_seq_len = max(args.max_seq_len,
                       max(len(x.prompt_ids) for x in items) + 64)
    print(f"eval max_seq_len={eval_seq_len}")
    engine = RolloutEngine(model, tokenizer, max_batch=1,
                           max_seq_len=eval_seq_len, kv_ratio=0.5,
                           device=device)
    policy = EnginePolicy(
        engine, tokenizer,
        Sampling(max_tokens=48, temperature=0.0), fewshot=fewshot)
    val_ds = MapDataset.load(Path(args.data) / "maps" / "val.jsonl")
    test_ds = MapDataset.load(Path(args.data) / "maps" / "test_id.jsonl")

    def pick(ds, limit):
        out = []
        for level in levels:
            pool = sorted((r for r in ds.records if r.level == level),
                          key=lambda r: r.hash)
            out.extend(pool[:limit])
        return out

    val = evaluate(policy, pick(val_ds, args.val_limit),
                   obs_style=args.obs_style,
                   memory=args.memory, show_visited=args.visited,
                   split="val")
    test = evaluate(policy, pick(test_ds, args.test_limit),
                    obs_style=args.obs_style,
                    memory=args.memory, show_visited=args.visited,
                    split="test")
    print(f"val: {val}")
    print(f"test: {test}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({
        "samples": len(items),
        "actions": dict(action_counts),
        "val": val,
        "test": test,
        "adapter": str(save_path),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
