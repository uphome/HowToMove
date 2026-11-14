"""12x12 网格迷宫环境。

坐标约定
    grid[y][x]，y 向下增长。
    up = y-1, down = y+1, left = x-1, right = x+1。

设计要点
    - 墙只负责"形状"，可解性与难度由 BFS 距离场裁决；
    - 距离场是从终点反向 BFS 得到的全图最短距离，供 step() 做
      potential-based 距离塑形（绕墙时不会先吃负奖励）；
    - 生成阶段就保证 dist[start] <= max_steps - 2，否则 +50 永远拿不到。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

WALL = "#"
FREE = "."
AGENT = "A"
GOAL = "G"

ACTIONS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

REWARD_FORMAT = -10.0
REWARD_COLLISION = -2.0
REWARD_TERMINAL = 50.0
REWARD_STEP = -0.1
REWARD_DISTANCE = 1.0

INF = 10**9
LEVELS = ("L0", "L1", "L2", "L3", "L4")

# detour = shortest_path - manhattan，在 4 邻域网格上恒为偶数
# （每多走一步"远离"就多 2）。所以 2 = 至少倒退 1 步，4 = 至少倒退 2 步。
LEVEL_RULES: dict[str, dict[str, int]] = {
    "L0": {"min_dist": 8, "min_detour": 0},
    "L1": {"min_dist": 8, "min_detour": 2},
    "L2": {"min_dist": 8, "min_detour": 2},
    "L3": {"min_dist": 8, "min_detour": 4},
    "L4": {"min_dist": 8, "min_detour": 2},
}


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_field(
    width: int,
    height: int,
    walls: frozenset[tuple[int, int]] | set[tuple[int, int]],
    goal: tuple[int, int],
) -> list[list[int]]:
    """从 goal 出发的 4 邻域 BFS 距离场，不可达为 INF。"""
    dist = [[INF] * width for _ in range(height)]
    gx, gy = goal
    if not (0 <= gx < width and 0 <= gy < height) or (gx, gy) in walls:
        return dist
    dist[gy][gx] = 0
    queue = deque([(gx, gy)])
    while queue:
        x, y = queue.popleft()
        nd = dist[y][x] + 1
        for dx, dy in ACTIONS.values():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < width
                and 0 <= ny < height
                and (nx, ny) not in walls
                and dist[ny][nx] == INF
            ):
                dist[ny][nx] = nd
                queue.append((nx, ny))
    return dist


# ---------------------------------------------------------------- 形状生成


def _vertical_wall(width: int, height: int, rng: random.Random) -> set[tuple[int, int]]:
    """竖墙 + 1 格门。"""
    x = rng.randint(2, width - 3)
    y0 = rng.randint(0, height - 3)
    y1 = rng.randint(y0 + 2, height - 1)
    door = rng.randint(y0, y1)
    return {(x, y) for y in range(y0, y1 + 1) if y != door}


def _l_wall(width: int, height: int, rng: random.Random) -> set[tuple[int, int]]:
    """L 型墙：一横一竖，四向随机。"""
    arm_x = rng.randint(2, 4)
    arm_y = rng.randint(2, 4)
    cx = rng.randint(1, width - 2)
    cy = rng.randint(1, height - 2)
    sx = rng.choice((-1, 1))
    sy = rng.choice((-1, 1))
    cells = {(cx + sx * i, cy) for i in range(arm_x + 1)}
    cells |= {(cx, cy + sy * j) for j in range(arm_y + 1)}
    return {
        (x, y) for (x, y) in cells if 0 <= x < width and 0 <= y < height
    }


def _rotate_cells(
    cells: set[tuple[int, int]], w: int, h: int, k: int
) -> tuple[set[tuple[int, int]], int, int]:
    """在 w x h 的局部包围盒内旋转 k*90 度，返回新格子与新宽高。"""
    if k % 2 == 0:
        nw, nh = w, h
    else:
        nw, nh = h, w
    out: set[tuple[int, int]] = set()
    for i, j in cells:
        if k % 4 == 0:
            ni, nj = i, j
        elif k % 4 == 1:
            ni, nj = h - 1 - j, i
        elif k % 4 == 2:
            ni, nj = w - 1 - i, h - 1 - j
        else:
            ni, nj = j, w - 1 - i
        out.add((ni, nj))
    return out, nw, nh


def _u_wall(width: int, height: int, rng: random.Random) -> set[tuple[int, int]]:
    """U/C 型墙：三面围死 + 一面留 1~2 格窄门。

    这是真正会逼模型"先远离目标"的形状，也是 BFS 距离奖励相对 Manhattan
    的唯一价值所在。
    """
    w = rng.randint(3, 5)
    h = rng.randint(3, 4)
    local: set[tuple[int, int]] = set()
    for i in range(w):
        local.add((i, h - 1))
    for j in range(h):
        local.add((0, j))
        local.add((w - 1, j))
    door_w = rng.randint(1, 2)
    door_start = rng.randint(0, w - door_w)
    for i in range(w):
        if not (door_start <= i < door_start + door_w):
            local.add((i, 0))

    local, w, h = _rotate_cells(local, w, h, rng.randint(0, 3))
    x0 = rng.randint(0, width - w)
    y0 = rng.randint(0, height - h)
    return {(x0 + i, y0 + j) for (i, j) in local}


_SHAPES = {
    "L1": (_vertical_wall,),
    "L2": (_l_wall,),
    "L3": (_u_wall,),
    "L4": (_vertical_wall, _l_wall, _u_wall),
}


def _make_walls(
    level: str, width: int, height: int, rng: random.Random
) -> set[tuple[int, int]]:
    if level == "L0":
        return set()
    pool = _SHAPES[level]
    n = 1 if level != "L4" else rng.randint(1, 3)
    walls: set[tuple[int, int]] = set()
    for _ in range(n):
        walls |= rng.choice(pool)(width, height, rng)
    return walls


def _place(
    walls: set[tuple[int, int]],
    width: int,
    height: int,
    max_steps: int,
    min_dist: int,
    min_detour: int,
    rng: random.Random,
):
    """在自由格中挑一对 (start, goal)，满足距离与绕行约束。

    一次性扫完所有候选，不靠盲重试。
    """
    free = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if (x, y) not in walls
    ]
    if len(free) < 2:
        return None
    goal = rng.choice(free)
    dist = bfs_field(width, height, walls, goal)
    cands = [
        (x, y)
        for (x, y) in free
        if min_dist <= dist[y][x] <= max_steps - 2
        and dist[y][x] - manhattan((x, y), goal) >= min_detour
    ]
    if not cands:
        return None
    start = rng.choice(cands)
    return start, goal, dist


# ---------------------------------------------------------------- Maze


@dataclass(frozen=True)
class Maze:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    max_steps: int
    level: str = "L0"
    seed: int | None = None
    dist: list[list[int]] = field(default_factory=list, repr=False, compare=False)

    def distance(self, pos: tuple[int, int]) -> int:
        return self.dist[pos[1]][pos[0]]

    @property
    def path_length(self) -> int:
        return self.distance(self.start)

    @property
    def detour(self) -> int:
        return self.path_length - manhattan(self.start, self.goal)

    def is_free(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.walls

    def shortest_path(self) -> list[tuple[int, int]]:
        """沿距离场下降的一条最短路（含起点与终点）。"""
        path = [self.start]
        pos = self.start
        d = self.distance(pos)
        while pos != self.goal:
            for dx, dy in ACTIONS.values():
                nxt = (pos[0] + dx, pos[1] + dy)
                if self.is_free(*nxt) and self.distance(nxt) == d - 1:
                    pos = nxt
                    d -= 1
                    path.append(pos)
                    break
            else:  # pragma: no cover - BFS 距离场保证不会发生
                raise RuntimeError(f"no descending neighbour from {pos}")
        return path

    def to_ascii(self, pos: tuple[int, int] | None = None) -> str:
        pos = self.start if pos is None else pos
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == pos:
                    row.append(AGENT)
                elif (x, y) == self.goal:
                    row.append(GOAL)
                elif (x, y) in self.walls:
                    row.append(WALL)
                else:
                    row.append(FREE)
            rows.append("".join(row))
        return "\n".join(rows)


def generate(
    level: str = "L0",
    *,
    width: int = 12,
    height: int = 12,
    max_steps: int = 30,
    seed: int | None = None,
    max_attempts: int = 200,
) -> Maze:
    """生成一张满足全部不变量的迷宫。"""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {LEVELS}")
    if max_steps < 4:
        raise ValueError("max_steps too small")
    rules = LEVEL_RULES[level]
    rng = random.Random(seed)

    for _ in range(max_attempts):
        walls = _make_walls(level, width, height, rng)
        placed = _place(
            walls,
            width,
            height,
            max_steps,
            rules["min_dist"],
            rules["min_detour"],
            rng,
        )
        if placed is None:
            continue
        start, goal, dist = placed
        return Maze(
            width=width,
            height=height,
            walls=frozenset(walls),
            start=start,
            goal=goal,
            max_steps=max_steps,
            level=level,
            seed=seed,
            dist=dist,
        )

    raise RuntimeError(f"failed to generate maze: level={level} seed={seed}")


# ---------------------------------------------------------------- Episode


@dataclass
class StepResult:
    pos: tuple[int, int]
    reward: float
    done: bool
    info: dict


class MazeEnv:
    """单条轨迹的可变状态：位置 / 步数 / 最近动作 / 是否结束。"""

    def __init__(self, maze: Maze, memory: int = 3):
        self.maze = maze
        self.memory = memory
        self.pos = maze.start
        self.steps = 0
        self.last_actions: list[str] = []
        self.done = False
        self.terminated = False
        self.truncated = False

    def reset(self) -> tuple[int, int]:
        self.pos = self.maze.start
        self.steps = 0
        self.last_actions = []
        self.done = False
        self.terminated = False
        self.truncated = False
        return self.pos

    def step(self, action: str) -> StepResult:
        if self.done:
            raise RuntimeError("episode already finished")
        maze = self.maze
        self.steps += 1
        reward = REWARD_STEP

        if action not in ACTIONS:
            reward += REWARD_FORMAT
            self._record("bad")
            return self._finish(reward, reason="bad_format")

        self._record(action)
        x, y = self.pos
        dx, dy = ACTIONS[action]
        nx, ny = x + dx, y + dy

        if not maze.is_free(nx, ny):
            reward += REWARD_COLLISION
            return self._after_step(reward, reason="collision")

        delta = maze.dist[y][x] - maze.dist[ny][nx]
        self.pos = (nx, ny)
        reward += REWARD_DISTANCE * delta

        if self.pos == maze.goal:
            reward += REWARD_TERMINAL
            return self._finish(reward, reason="goal")

        return self._after_step(reward, reason="move", delta=delta)

    def _record(self, action: str) -> None:
        if self.memory <= 0:
            return
        self.last_actions.append(action)
        if len(self.last_actions) > self.memory:
            del self.last_actions[: -self.memory]

    def _finish(self, reward: float, reason: str, **extra) -> StepResult:
        self.done = True
        self.terminated = True
        return StepResult(
            self.pos, reward, True, {"reason": reason, "steps": self.steps, **extra}
        )

    def _after_step(self, reward: float, reason: str, **extra) -> StepResult:
        if self.steps >= self.maze.max_steps:
            self.done = True
            self.truncated = True
            return StepResult(
                self.pos,
                reward,
                True,
                {
                    "reason": "truncated",
                    "last_reason": reason,
                    "steps": self.steps,
                    **extra,
                },
            )
        return StepResult(
            self.pos, reward, False, {"reason": reason, "steps": self.steps, **extra}
        )
