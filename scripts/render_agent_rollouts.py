"""Render greedy Agent rollouts for the unified GRPO checkpoint.

Produces one 2x5 trajectory figure per level and one GIF per level. Collision
attempts are marked explicitly; blue line is the executed path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from agent import ChatPolicy, MazeAgent  # noqa: E402
from dataset import load_records  # noqa: E402
from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from grpo.adapter import LoRAAdapter, tie_embeddings  # noqa: E402


def draw_base(ax, rec, *, show_title=True):
    maze = rec.to_maze()
    grid = np.ones((maze.height, maze.width, 3))
    for x, y in maze.walls:
        grid[y, x] = (0.16, 0.16, 0.16)
    ax.imshow(grid, origin="upper",
              extent=(-0.5, maze.width - 0.5,
                      maze.height - 0.5, -0.5))
    ax.set_xticks(range(maze.width))
    ax.set_yticks(range(maze.height))
    ax.grid(color="#d0d0d0", linewidth=0.35)
    ax.tick_params(labelbottom=False, labelleft=False,
                   bottom=False, left=False)
    ax.set_aspect("equal")
    ax.scatter([maze.start[0]], [maze.start[1]], s=42,
               color="#2f855a", edgecolor="white", zorder=4)
    ax.scatter([maze.goal[0]], [maze.goal[1]], s=52, marker="X",
               color="#c53030", edgecolor="white", zorder=4)
    if show_title:
        ax.set_title(f"{rec.id[:10]}", fontsize=7)
    return maze


def draw_run(ax, rec, run):
    maze = draw_base(ax, rec)
    positions = [maze.start] + [tuple(t["pos"]) for t in run.turns]
    ax.plot([p[0] for p in positions], [p[1] for p in positions],
            color="#2b6cb0", linewidth=1.7, alpha=0.9, zorder=3)
    collisions = 0
    for i, turn in enumerate(run.turns, 1):
        result = turn["tool_result"]
        if result["blocked"]:
            collisions += 1
            x, y = result["pos"]
            ax.scatter([x], [y], s=34, marker="x", color="#e53e3e",
                       linewidths=1.3, zorder=5)
        x, y = result["pos"]
        ax.text(x + 0.12, y - 0.12, str(i), fontsize=5,
                color="#1a365d", zorder=6)
    ax.set_title(
        f"{rec.id[:9]}  {'OK' if run.success else 'FAIL'}\n"
        f"steps={run.steps}  coll={collisions}",
        fontsize=6.5, pad=2,
    )
    return positions, collisions


def render_gif(rec, run, out_path: Path):
    maze = rec.to_maze()
    positions = [maze.start] + [tuple(t["pos"]) for t in run.turns]
    frames = []
    for i in range(len(positions)):
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        draw_base(ax, rec, show_title=False)
        ax.plot([p[0] for p in positions[:i + 1]],
                [p[1] for p in positions[:i + 1]],
                color="#2b6cb0", linewidth=2.0, alpha=0.9)
        x, y = positions[i]
        ax.scatter([x], [y], s=90, color="#dd6b20",
                   edgecolor="white", zorder=7)
        ax.set_title(f"{rec.id[:12]}  step {i}", fontsize=9)
        text = "start"
        if i > 0:
            turn = run.turns[i - 1]
            result = turn["tool_result"]
            text = (f"action={turn['action']}  "
                    f"ok={result['ok']}  blocked={result['blocked']}  "
                    f"done={result['done']}")
        fig.text(0.5, 0.02, text, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(frame))
        plt.close(fig)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=450, loop=0, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(paths.MODEL_PATH))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--per-level", type=int, default=5)
    ap.add_argument("--levels", default="L0,L1,L2,L3,L4")
    ap.add_argument("--ids-json", default=None,
                    help="optional {level: [map_id, ...]} selection")
    ap.add_argument("--out", default="reports/curves")
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model)
    tie_embeddings(model)
    blob = torch.load(args.checkpoint, map_location=device)
    adapter = LoRAAdapter(model, r=16, alpha=32)
    adapter.load_state_dict(blob.get("adapter", blob))
    model.eval()
    engine = RolloutEngine(model, tokenizer, max_batch=1,
                           max_seq_len=4096, kv_ratio=0.5, device=device)
    policy = ChatPolicy(engine, tokenizer,
                        Sampling(max_tokens=48, temperature=0.0))

    records = load_records(Path(args.data) / "maps" / f"{args.split}.jsonl")
    levels = [x.strip().upper() for x in args.levels.split(",") if x.strip()]
    ids_by_level = {}
    if args.ids_json:
        ids_by_level = json.loads(Path(args.ids_json).read_text(
            encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for level in levels:
        pool = sorted((r for r in records if r.level == level),
                      key=lambda r: r.hash)
        wanted = ids_by_level.get(level)
        if wanted:
            by_id = {r.id: r for r in pool}
            chosen = [by_id[x] for x in wanted if x in by_id]
        else:
            chosen = pool[:args.per_level]
        if not chosen:
            raise RuntimeError(f"no selected maps for {level}")
        runs = []
        for rec in chosen:
            agent = MazeAgent(rec.to_maze(), policy, history_window=12,
                              episode_id=f"viz/{rec.id}")
            run = agent.run()
            runs.append(run)
            print(f"{level} {rec.id[:12]} "
                  f"success={run.success} steps={run.steps} "
                  f"blocked={sum(1 for t in run.turns if t['tool_result']['blocked'])}")
        cols = min(5, len(chosen))
        rows = (len(chosen) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(3.2 * cols, 3.8 * rows))
        axes = np.asarray(axes).reshape(-1)
        for ax, rec, run in zip(axes, chosen, runs):
            draw_run(ax, rec, run)
        for ax in axes[len(chosen):]:
            ax.axis("off")
        fig.suptitle(f"GRPO chunk1 rollouts - {level} ({args.split})",
                     fontsize=11)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90,
                            bottom=0.03, wspace=0.08, hspace=0.32)
        png = out / f"grpo_chunk1_rollout_{level}.png"
        fig.savefig(png, dpi=160)
        plt.close(fig)
        print(f"png -> {png}")

        # Use a failed trajectory when available, otherwise the first one.
        idx = next((i for i, r in enumerate(runs) if not r.success), 0)
        gif = out / f"grpo_chunk1_rollout_{level}.gif"
        render_gif(chosen[idx], runs[idx], gif)
        print(f"gif -> {gif}")

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
