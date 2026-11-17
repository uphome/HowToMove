"""Run G independent agents on one map through one batched policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from agent import AgentGroup, ChatPolicy  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import LEVELS, generate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--level", default="L0", choices=LEVELS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--g", type=int, default=8)
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    maze = generate(level=args.level, seed=args.seed)
    print(f"maze={maze.level} seed={maze.seed} start={maze.start} "
          f"goal={maze.goal} shortest={maze.path_length} g={args.g}")

    model, tokenizer, device = load_model(args.model)
    engine = RolloutEngine(model, tokenizer, max_batch=args.g,
                           max_seq_len=args.max_seq_len,
                           kv_ratio=0.5, device=device)
    policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=args.max_tokens, temperature=args.temperature),
    )

    def on_step(index: int, turn: dict) -> None:
        result = turn["tool_result"]
        print(f"g{index} t{turn['turn']:02d} "
              f"{turn['action'] or 'invalid':7s} "
              f"pos={tuple(turn['pos'])} blocked={result['blocked']} "
              f"done={result['done']} reason={result['reason']}")

    group = AgentGroup(
        maze, policy, g=args.g,
        history_window=args.history_window,
        episode_prefix=f"agent-{args.level}-{args.seed}",
        on_step=on_step,
    )
    runs = group.run()
    print("\nsummary:")
    for run in runs:
        print(run.episode_id, run.summary())

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([
            {"summary": run.summary(),
             "messages": run.messages,
             "turns": run.turns}
            for run in runs
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"轨迹 → {out}")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
