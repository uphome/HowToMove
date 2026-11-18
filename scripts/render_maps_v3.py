"""Render ten maps per level from the map-only v3 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dataset import MapRecord, load_records  # noqa: E402


def render(ax, rec: MapRecord, *, show_title: bool = True) -> None:
    maze = rec.to_maze()
    grid = np.ones((maze.height, maze.width, 3))
    for x, y in maze.walls:
        grid[y, x] = (0.16, 0.16, 0.16)

    ax.imshow(grid, origin="upper",
              extent=(-0.5, maze.width - 0.5, maze.height - 0.5, -0.5))
    ax.set_xticks(range(maze.width))
    ax.set_yticks(range(maze.height))
    ax.grid(color="#d0d0d0", linewidth=0.35)
    ax.tick_params(labelbottom=False, labelleft=False,
                   bottom=False, left=False)
    ax.set_aspect("equal")

    path = maze.shortest_path()
    ax.plot([p[0] for p in path], [p[1] for p in path],
            color="#2b6cb0", linewidth=1.6, alpha=0.85)
    ax.scatter([maze.start[0]], [maze.start[1]], s=42,
               color="#2f855a", edgecolor="white", zorder=4)
    ax.scatter([maze.goal[0]], [maze.goal[1]], s=52, marker="X",
               color="#c53030", edgecolor="white", zorder=4)
    if show_title:
        ax.set_title(
            f"{rec.id[:10]}  shortest={rec.shortest}",
            fontsize=7,
            pad=2,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test",
                    choices=("train", "val", "test"))
    ap.add_argument("--per-level", type=int, default=10)
    ap.add_argument("--out", default="reports/curves")
    args = ap.parse_args()

    records = load_records(Path(args.data) / "maps" / f"{args.split}.jsonl")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    levels = ("L1", "L2", "L3", "L4")
    selected = {}
    for level in levels:
        pool = sorted((r for r in records if r.level == level),
                      key=lambda r: r.hash)
        selected[level] = pool[:args.per_level]

    # One readable 2x5 figure per level.
    for level in levels:
        fig, axes = plt.subplots(2, 5, figsize=(16, 7.6))
        for ax, rec in zip(axes.flat, selected[level]):
            render(ax, rec)
        for ax in axes.flat[len(selected[level]):]:
            ax.axis("off")
        fig.suptitle(f"{level} maps ({args.split}, first {args.per_level})",
                     fontsize=11)
        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.90, bottom=0.03,
            wspace=0.08, hspace=0.32,
        )
        path = out / f"maps_{level}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"{level} -> {path}")

    # Compact overview: four rows, ten columns. Keep titles off because the
    # panels are intentionally small; the per-level figures show details.
    fig, axes = plt.subplots(4, 10, figsize=(24, 10.8))
    for row, level in enumerate(levels):
        for col in range(10):
            ax = axes[row, col]
            if col < len(selected[level]):
                render(ax, selected[level][col], show_title=False)
            else:
                ax.axis("off")
        fig.text(0.005, 1 - (row + 0.5) / 4, level,
                 rotation=90, va="center", fontsize=11)
    fig.suptitle(f"Map samples ({args.split})", fontsize=13)
    fig.subplots_adjust(
        left=0.035, right=0.99, top=0.93, bottom=0.03,
        wspace=0.04, hspace=0.12,
    )
    path = out / "maps_L1_L4_overview.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"overview -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
