"""把数据集里的几张地图渲染出来看。

    python scripts/render_dataset.py
    python scripts/render_dataset.py --per-sheet 6 --cell 22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import MapDataset  # noqa: E402
from env.render import render_state  # noqa: E402
from env.viz import render_png, render_sheet  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--cell", type=int, default=26)
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    train = MapDataset.load(root / "maps/train.jsonl")
    hard = MapDataset.load(root / "maps/test_hard.jsonl")
    ood = MapDataset.load(root / "maps/test_ood.jsonl")

    made = []

    # 1) 每个级别一张
    picks = []
    for level in ("L0", "L1", "L2", "L3", "L4"):
        pool = train.filter(level=level)
        if len(pool):
            picks.append((level, pool[0].to_maze()))
    made.append(render_sheet(picks, out / "ds_levels.png", cell=args.cell, cols=5))

    # 2) L3 六张
    l3 = train.filter(level="L3").records[: args.per_sheet]
    made.append(render_sheet([(r.id, r.to_maze()) for r in l3],
                             out / "ds_L3.png", cell=args.cell, cols=3))

    # 3) hard（detour>=6）
    made.append(render_sheet([(r.id, r.to_maze()) for r in
                              hard.records[: args.per_sheet]],
                             out / "ds_hard.png", cell=args.cell, cols=3))

    # 4) OOD：8x8 与 16x16 分开渲染（尺寸不同，混在一起比例会乱）
    small = [r for r in ood.records if r.width == 8][:4]
    large = [r for r in ood.records if r.width == 16][:4]
    made.append(render_sheet([(r.id, r.to_maze()) for r in small],
                             out / "ds_ood_8.png", cell=34, cols=4))
    made.append(render_sheet([(r.id, r.to_maze()) for r in large],
                             out / "ds_ood_16.png", cell=26, cols=4))

    # 5) 单张样例（带最短路上色）
    demo = train.filter(level="L3").records[0]
    maze = demo.to_maze()
    made.append(render_png(maze, out / "ds_sample.png", cell=34,
                           path=maze.shortest_path(), pos=maze.start))

    for p in made:
        print(p)

    print("\n模型看到的观测（L3 样例）：")
    print("-" * 60)
    print(render_state(maze, maze.start))
    print("-" * 60)
    print(f"shortest={demo.shortest} detour={demo.detour} walls={len(demo.walls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
