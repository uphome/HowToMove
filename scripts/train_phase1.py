"""阶段一实验：单步 GRPO，能不能自己学会「delta → 方向」？

不依赖 SFT。每次更新：
    采样 K 个状态 → 每个状态 G 条单步轨迹 → 组内标准化 → GRPO 更新

关键观察指标：
    - 第 0 步 ratio 是否 ≈ 1（训推一致性，T3）
    - 组内退化比例（全一样 → 无梯度）
    - 固定 eval 集上的 optimal 率随更新变化

    python scripts/train_phase1.py --updates 20 --states 4 --g 8
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

from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import (  # noqa: E402
    ACTIONS, LEVELS, REWARD_COLLISION, REWARD_DISTANCE, REWARD_FORMAT,
    REWARD_STEP, REWARD_TERMINAL, generate,
)
from env.render import render_state  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402
from grpo.train import GRPOTrainer, Sample, TrainConfig, group_advantages  # noqa: E402
from prompts import render_prompt  # noqa: E402
from rollout import parse_action  # noqa: E402

def one_step_reward(maze, pos, action):
    if action is None:
        return REWARD_FORMAT + REWARD_STEP, "invalid"
    dx, dy = ACTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not maze.is_free(*nxt):
        return REWARD_COLLISION + REWARD_STEP, "blocked"
    delta = maze.distance(pos) - maze.distance(nxt)
    r = REWARD_DISTANCE * delta + REWARD_STEP
    if nxt == maze.goal:
        return r + REWARD_TERMINAL, "goal"
    return r, "move" if delta else "side"


def sample_states(n: int, rng: random.Random, offset: int = 0):
    out = []
    i = offset
    while len(out) < n:
        level = LEVELS[i % len(LEVELS)]
        maze = generate(level=level, seed=rng.randrange(10**6) + i)
        i += 1
        cand = [(x, y) for y in range(maze.height) for x in range(maze.width)
                if maze.is_free(x, y) and maze.distance((x, y)) >= 3]
        if cand:
            out.append((maze, rng.choice(cand)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--states", type=int, default=4, help="每次更新的状态数")
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--obs-style", default="minimal")
    ap.add_argument("--eval-states", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", default="out/lora_phase1.pt")
    ap.add_argument("--json", default="out/phase1_log.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    eval_states = sample_states(args.eval_states, random.Random(1234))

    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    model.eval()
    tie_embeddings(model)

    def prompt_ids(maze, pos):
        state = render_state(maze, pos, style=args.obs_style)
        _, ids = render_prompt(tokenizer, state)
        return ids

    def evaluate(engine):
        acts, rewards, kinds = [], [], []
        for i in range(0, len(eval_states), args.g):
            batch = eval_states[i:i + args.g]
            prompts = [prompt_ids(m, p) for m, p in batch]
            outs = engine.generate_batch(prompts, Sampling(
                max_tokens=args.max_tokens, temperature=0.0))
            for (m, p), o in zip(batch, outs):
                a = parse_action(tokenizer.decode(o.token_ids, skip_special_tokens=True))
                r, k = one_step_reward(m, p, a)
                acts.append(a); rewards.append(r); kinds.append(k)
        return {
            "valid": sum(a is not None for a in acts) / len(acts),
            "good": sum(k in ("move", "goal") for k in kinds) / len(kinds),
            "blocked": sum(k == "blocked" for k in kinds) / len(kinds),
            "invalid": sum(k == "invalid" for k in kinds) / len(kinds),
            "reward": statistics.mean(rewards),
        }

    engine = RolloutEngine(model, tokenizer, max_batch=args.g,
                           max_seq_len=768, kv_ratio=0.5, device=device)
    adapter = LoRAAdapter(model, r=args.lora_r, alpha=2 * args.lora_r)
    n_train = sum(p.numel() for p in adapter.trainable_parameters())
    print(f"LoRA params: {n_train/1e6:.2f}M  (r={args.lora_r})")
    trainer = GRPOTrainer(adapter, TrainConfig(
        lr=args.lr, clip_eps=args.clip, beta=args.beta))

    base = evaluate(engine)
    print(f"[eval @0] {base}")
    hist = [{"update": 0, **base}]

    t_start = time.time()
    skipped = 0
    for u in range(1, args.updates + 1):
        states = sample_states(args.states, rng, offset=10**5 + u * 100)
        samples: list[Sample] = []
        group_std = []
        for maze, pos in states:
            ids = prompt_ids(maze, pos)
            outs = engine.generate_batch(
                [ids] * args.g,
                Sampling(max_tokens=args.max_tokens, temperature=args.temperature))
            rewards, rows = [], []
            for o in outs:
                a = parse_action(tokenizer.decode(o.token_ids,
                                                  skip_special_tokens=True))
                r, _ = one_step_reward(maze, pos, a)
                rewards.append(r)
                rows.append(o)
            advs, degen = group_advantages(rewards)
            group_std.append(statistics.pstdev(rewards))
            if degen:
                skipped += 1
                continue
            for o, adv in zip(rows, advs):
                if not o.token_ids:
                    continue
                samples.append(Sample(ids, list(o.token_ids),
                                      list(o.logprobs), float(adv)))

        if not samples:
            print(f"[{u:3d}] 全组退化，跳过")
            continue

        stats = trainer.update(samples)
        line = (f"[{u:3d}] n={len(samples):2d} skip={skipped} "
                f"loss={stats.loss:+.3f} pg={stats.pg:+.3f} kl={stats.kl:+.4f} "
                f"ratio=[{stats.ratio_min:.4f},{stats.ratio_max:.4f}] "
                f"clip={stats.clip_frac:.2f} drop={stats.drop_frac:.2f} "
                f"gstd={statistics.mean(group_std):.2f} "
                f"gnorm={stats.grad_norm:.3f}")
        print(line)
        if u % args.eval_every == 0 or u == args.updates:
            ev = evaluate(engine)
            print(f"      [eval @{u}] {ev}")
            hist.append({"update": u, **ev})

    dt = time.time() - t_start
    print(f"\n总耗时 {dt/60:.1f} min，退化组 {skipped}")
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.save)
    Path(args.json).write_text(json.dumps(hist, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(f"adapter → {args.save}\n日志 → {args.json}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
