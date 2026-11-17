"""Run one explicit tool-calling agent episode on a maze.

    python scripts/agent_play.py --level L0 --seed 7 --fewshot rel
    python scripts/agent_play.py --level L1 --seed 7 --fewshot rel_l1 \
        --ui-url http://127.0.0.1:7861
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from agent import ChatPolicy, MazeAgent  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import LEVELS, generate  # noqa: E402
from env.ui import VizClient, build_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--level", default="L0", choices=LEVELS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obs-style", default="rel")
    ap.add_argument("--fewshot", default="rel",
                    choices=["none", "rel", "rel_l1"])
    ap.add_argument("--history-window", type=int, default=12)
    ap.add_argument("--memory", type=int, default=3)
    ap.add_argument("--no-visited", action="store_false", dest="visited",
                    help="不把 visited_rel 放进 tool result")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--ui-url", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.fewshot != "none" and args.obs_style != "rel":
        ap.error("--fewshot 需要 --obs-style rel")

    maze = generate(level=args.level, seed=args.seed)
    print(f"maze={maze.level} seed={maze.seed} "
          f"start={maze.start} goal={maze.goal} shortest={maze.path_length}")

    ui = None
    if args.ui_url:
        client = VizClient(args.ui_url)
        if client.health():
            ui = client
            print(f"UI: {args.ui_url}")
        else:
            print(f"UI: {args.ui_url} 不可用，只打印轨迹")

    print(f"加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model)
    engine = RolloutEngine(model, tokenizer, max_batch=1,
                           max_seq_len=args.max_seq_len,
                           kv_ratio=0.5, device=device)
    policy = ChatPolicy(
        engine, tokenizer,
        Sampling(max_tokens=args.max_tokens, temperature=args.temperature),
    )

    def on_step(turn: dict) -> None:
        print(f"[{turn['turn']:2d}] {turn['action']:7s} "
              f"pos={tuple(turn['pos'])} reward={turn['reward']:+.2f} "
              f"reason={turn['info'].get('reason')}")
        if ui is None:
            return
        info = {
            "action": turn["action"],
            "reward": turn["reward"],
            **turn["info"],
        }
        try:
            ui.publish(build_frame(
                maze, tuple(turn["pos"]), path=None, mode="live",
                info=info,
                llm={
                    "text": turn["assistant_text"],
                    "action": turn["action"],
                    "logprob": turn["logprob"],
                },
                prompt=policy.last_prompt_text,
                episode=f"agent-{args.level}-{args.seed}",
                turn=turn["turn"],
            ))
        except Exception as exc:  # UI must not break the agent
            print(f"[viz] push failed: {exc}")

    agent = MazeAgent(
        maze, policy,
        obs_style=args.obs_style,
        fewshot=None if args.fewshot == "none" else args.fewshot,
        history_window=args.history_window,
        memory=args.memory,
        show_visited=args.visited,
        on_step=on_step,
    )
    run = agent.run()
    print("summary:", run.summary())
    print(f"messages={len(run.messages)} turns={run.steps}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "summary": run.summary(),
            "messages": run.messages,
            "turns": run.turns,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"轨迹 → {out}")
    engine.close()
    return 0 if run.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
