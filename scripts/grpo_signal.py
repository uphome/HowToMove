"""GRPO 可学性探针：起始策略在组内到底有没有"正确动作"？

GRPO 是信用分配，不是搜索。它只能在**已经采样到**的动作里提高概率。
所以做 GRPO 之前必须先回答一个可量化的问题：

    同一个状态采样 G 条，组内出现"能缩短距离的动作"的概率是多少？

如果这个概率接近 0，组内优势全为 0（或全是同一动作），梯度恒为零，
GRPO 再跑一万步也学不会 —— 这不是超参问题，是覆盖问题。

    python scripts/grpo_signal.py --states 20 --g 8
"""

from __future__ import annotations

import argparse
import collections
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import (  # noqa: E402
    ACTIONS, LEVELS, REWARD_COLLISION, REWARD_DISTANCE, REWARD_FORMAT,
    REWARD_STEP, REWARD_TERMINAL, generate,
)
from env.render import render_state  # noqa: E402
from prompts import render_prompt  # noqa: E402
from rollout import parse_action  # noqa: E402

def sample_states(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < n:
        level = LEVELS[i % len(LEVELS)]
        maze = generate(level=level, seed=seed * 5000 + i)
        i += 1
        cand = [(x, y) for y in range(maze.height) for x in range(maze.width)
                if maze.is_free(x, y) and maze.distance((x, y)) >= 3]
        if cand:
            out.append((maze, rng.choice(cand)))
    return out


def one_step_reward(maze, pos, action: str | None):
    if action is None:
        return REWARD_FORMAT + REWARD_STEP, "invalid"
    dx, dy = ACTIONS[action]
    nxt = (pos[0] + dx, pos[1] + dy)
    if not maze.is_free(*nxt):
        return REWARD_COLLISION + REWARD_STEP, "blocked"
    delta = maze.distance(pos) - maze.distance(nxt)
    r = REWARD_DISTANCE * delta + REWARD_STEP
    if nxt == maze.goal:
        r += REWARD_TERMINAL
        return r, "goal"
    return r, "move" if delta else "side"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--states", type=int, default=20)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--temps", default="0.7,1.0,1.2,1.5")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    temps = [float(t) for t in args.temps.split(",")]
    states = sample_states(args.states, args.seed)
    model, tokenizer, device = load_model(args.model)
    engine = RolloutEngine(model, tokenizer, max_batch=args.g, max_seq_len=1024,
                           kv_ratio=0.5, device=device)

    def run_group(maze, pos, temperature: float) -> list:
        state = render_state(maze, pos, [], style="minimal")
        _, ids = render_prompt(tokenizer, state)
        samples = engine.generate_batch(
            [ids] * args.g, Sampling(max_tokens=40, temperature=temperature))
        acts = [parse_action(tokenizer.decode(s.token_ids, skip_special_tokens=True))
                for s in samples]
        return acts

    print(f"states={len(states)}  G={args.g}  temps={temps}")
    print()
    print(f"  {'T':>4} {'all-same':>9} {'>=1 optimal':>12} {'>=1 closer':>11} "
          f"{'mean std':>9} {'zero-grad':>10}")
    for T in temps:
        all_same = has_opt = has_closer = 0
        stds = []
        for maze, pos in states:
            acts = run_group(maze, pos, T)
            rewards = [one_step_reward(maze, pos, a)[0] for a in acts]
            kinds = [one_step_reward(maze, pos, a)[1] for a in acts]
            if len(set(acts)) == 1:
                all_same += 1
            if any(k in ("move", "goal") for k in kinds):
                has_closer += 1
                has_opt += 1
            stds.append(statistics.pstdev(rewards))
        n = len(states)
        zero = sum(1 for v in stds if v < 1e-6)
        print(f"  {T:>4} {all_same:>9} {has_opt:>12} {has_closer:>11} "
              f"{statistics.mean(stds):>9.3f} {zero:>10}")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
