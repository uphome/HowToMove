"""观测编码消融：grid / coords / rows / bitmap × 历史长度。

回答两个问题：
  1. 只给起点、终点、障碍坐标（项目书原始设计）是否比 12x12 网格更好？
  2. 模型需不需要"上一步"的信息？

    python scripts/obs_ablation.py --n 30
    python scripts/obs_ablation.py --styles rows --history 0,1,3
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import ACTIONS, LEVELS, generate  # noqa: E402
from env.render import STYLES, render_state  # noqa: E402
from rollout import parse_action  # noqa: E402


def sample_states(n: int, seed: int = 0) -> list:
    """每个级别均匀采样，位置取"距离目标 >= 3"的自由格（避免开局就一格到终点）。"""
    rng = random.Random(seed)
    out = []
    i = 0
    while len(out) < n:
        level = LEVELS[i % len(LEVELS)]
        maze = generate(level=level, seed=seed * 1000 + i)
        i += 1
        cand = [
            (x, y)
            for y in range(maze.height)
            for x in range(maze.width)
            if maze.is_free(x, y)
            and maze.distance((x, y)) >= 3
            and (x, y) != maze.goal
        ]
        if not cand:
            continue
        out.append((maze, rng.choice(cand)))
    return out


def evaluate(model, tokenizer, engine, states, style: str, history: int) -> dict:
    from prompts import render_prompt

    stats = collections.Counter()
    tok_count = 0
    preds = []
    for maze, pos in states:
        state = render_state(maze, pos, ["right"] * history if history else [],
                             style=style)
        _, ids = render_prompt(tokenizer, state)
        tok_count += len(ids)
        sample = engine.generate(ids, Sampling(max_tokens=40, temperature=0.0))
        action = parse_action(tokenizer.decode(sample.token_ids,
                                               skip_special_tokens=True))
        preds.append(action)
        if action is None:
            stats["invalid"] += 1
            continue
        stats["valid"] += 1
        d0 = maze.distance(pos)
        dx, dy = ACTIONS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not maze.is_free(*nxt):
            stats["blocked"] += 1
        elif maze.distance(nxt) == d0 - 1:
            stats["optimal"] += 1
        elif maze.distance(nxt) < d0:
            stats["closer"] += 1
        else:
            stats["farther"] += 1
    stats["prompt_tokens"] = round(tok_count / len(states), 1)
    stats["preds"] = preds
    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--history", default="3", help="逗号分隔，如 0,1,3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    histories = [int(h) for h in args.history.split(",") if h.strip()]
    states = sample_states(args.n, args.seed)
    print(f"states={len(states)}  styles={styles}  history={histories}")

    model, tokenizer, device = load_model(args.model)
    engine = RolloutEngine(model, tokenizer, max_batch=1, max_seq_len=1024,
                           kv_ratio=0.5, device=device)
    try:
        rows = []
        for style in styles:
            for h in histories:
                r = evaluate(model, tokenizer, engine, states, style, h)
                n = len(states)
                rows.append((style, h, r))
                print(f"{style:8s} hist={h}  "
                      f"valid={r.get('valid', 0):2d}/{n}  "
                      f"optimal={r.get('optimal', 0):2d}  "
                      f"closer={r.get('closer', 0):2d}  "
                      f"blocked={r.get('blocked', 0):2d}  "
                      f"farther={r.get('farther', 0):2d}  "
                      f"invalid={r.get('invalid', 0):2d}  "
                      f"prompt~{r['prompt_tokens']}tok")
                print(f"         preds={r['preds']}")
        best = max(rows, key=lambda x: (x[2].get("optimal", 0) + x[2].get("closer", 0),
                                         x[2].get("valid", 0)))
        print(f"\nbest: style={best[0]} history={best[1]} "
              f"optimal+closer={best[2].get('optimal', 0) + best[2].get('closer', 0)}"
              f"/{len(states)}")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
