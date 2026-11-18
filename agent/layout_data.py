"""Load Layout + Task records for agent SFT/evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from env.maze import Maze, bfs_field


@dataclass
class TaskItem:
    task_id: str
    layout_id: str
    level: str
    start: tuple[int, int]
    goal: tuple[int, int]
    shortest: int
    detour: int
    max_steps: int
    walls: frozenset[tuple[int, int]]
    maze: Maze

    @property
    def id(self) -> str:
        return self.task_id

    @property
    def hash(self) -> str:
        return self.task_id

    def to_maze(self) -> Maze:
        return self.maze


def load_tasks(data_dir: str | Path, split: str) -> list[TaskItem]:
    root = Path(data_dir)
    layouts = {}
    with (root / "layouts" / f"{split}.jsonl").open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                layouts[r["layout_id"]] = r
    out = []
    with (root / "tasks" / f"{split}.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            layout = layouts.get(r["layout_id"])
            if layout is None:
                raise RuntimeError(
                    f"task {r['task_id']} references missing layout "
                    f"{r['layout_id']}")
            walls = frozenset(tuple(p) for p in layout["walls"])
            start = tuple(r["start"])
            goal = tuple(r["goal"])
            dist = bfs_field(layout["width"], layout["height"], walls, goal)
            maze = Maze(
                width=int(layout["width"]),
                height=int(layout["height"]),
                walls=walls,
                start=start,
                goal=goal,
                max_steps=int(layout["max_steps"]),
                level=r["level"],
                seed=None,
                dist=dist,
            )
            out.append(TaskItem(
                task_id=r["task_id"],
                layout_id=r["layout_id"],
                level=r["level"],
                start=start,
                goal=goal,
                shortest=int(r["shortest"]),
                detour=int(r["detour"]),
                max_steps=int(layout["max_steps"]),
                walls=walls,
                maze=maze,
            ))
    return out


__all__ = ["TaskItem", "load_tasks"]
