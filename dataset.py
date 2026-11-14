"""数据集：地图记录的读写、重建、采样、单步最优动作。

设计约束（见 reports/dataset.md）：
  - 地图以 (w, h, walls, start, goal) 的规范 JSON 哈希为唯一标识；
  - 分桶按内容哈希，保证同一张图不会跨 split；
  - 存显式坐标而不是 (level, seed)，与生成器版本解耦；
  - BFS 距离场在加载时重算（O(144)），不进文件。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from env.maze import ACTIONS, Maze, bfs_field, generate, manhattan

SPLIT_BUCKETS = (
    (0, 70, "train"),
    (70, 84, "val"),
    (84, 92, "test_id"),
    (92, 100, "test_hard"),
)
SPLITS = tuple(name for _, _, name in SPLIT_BUCKETS)


def canonical_payload(width: int, height: int, walls: Iterable, start, goal) -> str:
    return json.dumps(
        {
            "w": int(width),
            "h": int(height),
            "walls": sorted([int(x), int(y)] for x, y in walls),
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(width: int, height: int, walls: Iterable, start, goal) -> str:
    return hashlib.sha256(
        canonical_payload(width, height, walls, start, goal).encode()
    ).hexdigest()


def split_for_hash(h: str) -> str:
    bucket = int(h[:8], 16) % 100
    for lo, hi, name in SPLIT_BUCKETS:
        if lo <= bucket < hi:
            return name
    raise AssertionError(bucket)


@dataclass(frozen=True)
class MapRecord:
    id: str
    level: str
    seed: int | None
    width: int
    height: int
    max_steps: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    shortest: int
    detour: int
    hash: str

    # ---- 构造 ----

    @classmethod
    def from_maze(cls, maze: Maze, *, ident: str | None = None,
                  seed: int | None = None) -> "MapRecord":
        h = content_hash(maze.width, maze.height, maze.walls, maze.start, maze.goal)
        return cls(
            id=ident or f"{maze.level}-{h[:10]}",
            level=maze.level,
            seed=maze.seed if seed is None else seed,
            width=maze.width,
            height=maze.height,
            max_steps=maze.max_steps,
            walls=frozenset(maze.walls),
            start=maze.start,
            goal=maze.goal,
            shortest=maze.path_length,
            detour=maze.detour,
            hash=h,
        )

    # ---- 重建 ----

    def to_maze(self) -> Maze:
        dist = bfs_field(self.width, self.height, self.walls, self.goal)
        return Maze(
            width=self.width,
            height=self.height,
            walls=self.walls,
            start=self.start,
            goal=self.goal,
            max_steps=self.max_steps,
            level=self.level,
            seed=self.seed,
            dist=dist,
        )

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "level": self.level,
            "seed": self.seed,
            "w": self.width,
            "h": self.height,
            "max_steps": self.max_steps,
            "walls": sorted([int(x), int(y)] for x, y in self.walls),
            "start": list(self.start),
            "goal": list(self.goal),
            "shortest": self.shortest,
            "detour": self.detour,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MapRecord":
        return cls(
            id=d["id"],
            level=d["level"],
            seed=d.get("seed"),
            width=d["w"],
            height=d["h"],
            max_steps=d.get("max_steps", 30),
            walls=frozenset((int(x), int(y)) for x, y in d["walls"]),
            start=(int(d["start"][0]), int(d["start"][1])),
            goal=(int(d["goal"][0]), int(d["goal"][1])),
            shortest=d["shortest"],
            detour=d["detour"],
            hash=d["hash"],
        )


# ---------------------------------------------------------------- I/O


def save_records(records: Iterable[MapRecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False,
                               separators=(",", ":")) + "\n")
            n += 1
    return n


def load_records(path: str | Path) -> list[MapRecord]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(MapRecord.from_dict(json.loads(line)))
    return out


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 采样


class MapDataset:
    def __init__(self, records: list[MapRecord]):
        self.records = list(records)

    @classmethod
    def load(cls, path: str | Path) -> "MapDataset":
        return cls(load_records(path))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[MapRecord]:
        return iter(self.records)

    def __getitem__(self, i) -> MapRecord:
        return self.records[i]

    def filter(self, *, level: str | None = None,
               min_detour: int | None = None) -> "MapDataset":
        out = self.records
        if level is not None:
            out = [r for r in out if r.level == level]
        if min_detour is not None:
            out = [r for r in out if r.detour >= min_detour]
        return MapDataset(out)

    def sample(self, k: int, rng, *, level: str | None = None,
               levels: Iterable[str] | None = None) -> list[MapRecord]:
        pool = self.records
        if level is not None:
            pool = [r for r in pool if r.level == level]
        if levels is not None:
            allowed = tuple(levels)
            pool = [r for r in pool if r.level in allowed]
        if not pool:
            raise ValueError("empty pool")
        if k > len(pool):
            raise ValueError(f"k={k} > pool={len(pool)}")
        return rng.sample(pool, k)

    def levels(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.level] = counts.get(r.level, 0) + 1
        return dict(sorted(counts.items()))


# ---------------------------------------------------------------- 工具


def optimal_actions(maze: Maze, pos: tuple[int, int]) -> list[str]:
    """所有能把 BFS 距离减 1 的动作（开阔区域会有并列）。"""
    d = maze.distance(pos)
    out = []
    for name, (dx, dy) in ACTIONS.items():
        nxt = (pos[0] + dx, pos[1] + dy)
        if maze.is_free(*nxt) and maze.distance(nxt) == d - 1:
            out.append(name)
    return out


def state_positions(maze: Maze, k: int, rng, *, min_dist: int = 3) -> list[tuple[int, int]]:
    """取 k 个「距终点 >= min_dist」的自由格（含起点）。"""
    cand = [(x, y) for y in range(maze.height) for x in range(maze.width)
            if maze.is_free(x, y) and maze.distance((x, y)) >= min_dist]
    if not cand:
        return [maze.start]
    if len(cand) <= k:
        return cand
    return rng.sample(cand, k)


def build_eval_states(records: list[MapRecord], k: int, seed: int = 0) -> list[dict]:
    """固定一批单步评测状态（不参与训练）。"""
    import random

    from env.render import render_state

    rng = random.Random(seed)
    picks = records[: k]
    out = []
    for i, rec in enumerate(picks):
        maze = rec.to_maze()
        pos = state_positions(maze, 1, rng, min_dist=3)[0]
        out.append({
            "id": f"es-{i:04d}",
            "map_id": rec.id,
            "level": rec.level,
            "pos": list(pos),
            "state": render_state(maze, pos),
            "optimal_actions": optimal_actions(maze, pos),
            "distance": maze.distance(pos),
            "manhattan": manhattan(pos, maze.goal),
        })
    return out


__all__ = [
    "SPLITS", "MapRecord", "MapDataset", "build_eval_states",
    "canonical_payload", "content_hash", "load_json", "load_records",
    "optimal_actions", "save_json", "save_records", "split_for_hash",
    "state_positions",
]
