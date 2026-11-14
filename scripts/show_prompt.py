"""打印模型实际看到的完整 prompt（含 tools 渲染）。

    python scripts/show_prompt.py --level L3 --seed 114
    python scripts/show_prompt.py --style coords --memory 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401

from transformers import AutoTokenizer  # noqa: E402

from env.maze import generate  # noqa: E402
from env.render import STYLES, render_state  # noqa: E402
from prompts import build_messages, render_prompt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--level", default="L3")
    ap.add_argument("--seed", type=int, default=114)
    ap.add_argument("--style", default="minimal", choices=list(STYLES))
    ap.add_argument("--memory", type=int, default=3)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--show-tokens", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    maze = generate(level=args.level, seed=args.seed)
    history = ["up", "left"][: args.memory] if args.memory else []
    state = render_state(maze, maze.start, history, style=args.style)

    text, ids = render_prompt(tokenizer, state, args.thinking)

    print("=" * 78)
    print(f"state  ({len(state)} chars)")
    print("=" * 78)
    print(state)
    print()
    print("=" * 78)
    print(f"prompt  ({len(ids)} tokens)   enable_thinking={args.thinking}")
    print("=" * 78)
    print(repr(text))
    print()
    print("-" * 78)
    print("可读形式：")
    print("-" * 78)
    print(text.replace("<|im_start|>", "\n<|im_start|>")
               .replace("<|im_end|>", "<|im_end|>\n"))
    if args.show_tokens:
        print("-" * 78)
        print("token ids:", ids)
        for t in ids:
            print(f"  {t:>7} {tokenizer.decode([t])!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
