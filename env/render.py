"""把迷宫状态渲染成文本观测。

四种风格（M1 实测 Qwen3-0.6B 接地能力很弱，需要挑信息密度合适的编码）：

    grid    12 行 ASCII 网格（默认）
    coords  起点/终点/当前 + 障碍坐标列表（项目书原始设计）
    rows    障碍按行分组、x 连续段压缩（最省 token）
    bitmap  12 行 0/1 占用图 + 独立坐标行

所有风格都带 `delta=(dx,dy)`：实测 12x12 网格会稀释模型对坐标的注意力，
显式 delta 把工具调用合法率从 4/8 提到 8/8。
"""

from __future__ import annotations

from .maze import AGENT, FREE, GOAL, WALL, Maze

STYLES = ("minimal", "coords", "rows", "grid", "bitmap")

HEADER_COORD = "maze {w}x{h}, x=right y=down"


def _delta(maze: Maze, pos: tuple[int, int]) -> tuple[int, int]:
    return maze.goal[0] - pos[0], maze.goal[1] - pos[1]


def _coord_line(maze: Maze, pos: tuple[int, int]) -> str:
    dx, dy = _delta(maze, pos)
    return (f"pos=({pos[0]},{pos[1]}) goal=({maze.goal[0]},{maze.goal[1]})"
            f" delta=({dx:+d},{dy:+d})")


def _last_line(last_actions) -> str:
    return "last=" + (",".join(last_actions) if last_actions else "none")


def render_grid(
    maze: Maze,
    pos: tuple[int, int],
    visited: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> list[str]:
    seen = set(visited)
    rows: list[str] = []
    for y in range(maze.height):
        cells: list[str] = []
        for x in range(maze.width):
            if (x, y) == pos:
                cells.append(AGENT)
            elif (x, y) == maze.goal:
                cells.append(GOAL)
            elif (x, y) in maze.walls:
                cells.append(WALL)
            elif (x, y) in seen:
                cells.append(",")
            else:
                cells.append(FREE)
        rows.append("".join(cells))
    return rows


def _wall_cells(maze: Maze) -> list[tuple[int, int]]:
    return sorted(maze.walls, key=lambda p: (p[1], p[0]))


def _runs(xs: list[int]) -> str:
    xs = sorted(xs)
    out: list[str] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def _rows_block(maze: Maze) -> str:
    by_row: dict[int, list[int]] = {}
    for x, y in maze.walls:
        by_row.setdefault(y, []).append(x)
    parts = [f"y={y} x={_runs(xs)}" for y, xs in sorted(by_row.items())]
    return "walls: " + ("; ".join(parts) if parts else "none")


def _coords_block(maze: Maze) -> str:
    cells = " ".join(f"({x},{y})" for x, y in _wall_cells(maze))
    return f"walls ({len(maze.walls)}): " + (cells or "none")


def _bitmap_block(maze: Maze) -> list[str]:
    return [
        "".join("1" if (x, y) in maze.walls else "0" for x in range(maze.width))
        for y in range(maze.height)
    ]


def _minimal_block(maze: Maze, pos: tuple[int, int]) -> str:
    """只有三样东西：当前位置、出口、障碍物坐标。没有表头、没有网格、没有 delta。"""
    cells = " ".join(f"({x},{y})" for x, y in _wall_cells(maze))
    return (f"pos=({pos[0]},{pos[1]}) goal=({maze.goal[0]},{maze.goal[1]})"
            "\n"
            f"walls: {cells or 'none'}")


def render_state(
    maze: Maze,
    pos: tuple[int, int],
    last_actions: list[str] | tuple[str, ...] = (),
    *,
    style: str = "minimal",
    visited: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> str:
    """状态块（不含 system prompt / 工具定义）。"""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}, expected one of {STYLES}")

    if style == "minimal":
        return _minimal_block(maze, pos)

    coord = _coord_line(maze, pos)
    tail = _last_line(last_actions)

    if style == "grid":
        lines = [
            f"maze ({maze.width}x{maze.height}, #=wall .=free A=you G=goal"
            f"{', =visited' if visited else ''}, x=right y=down):"
        ]
        lines.extend(render_grid(maze, pos, visited))
        lines.append(coord)
        lines.append(tail)
        return "\n".join(lines)

    if style == "bitmap":
        lines = [HEADER_COORD.format(w=maze.width, h=maze.height) + " (1=wall)"]
        lines.extend(_bitmap_block(maze))
        lines.append(coord)
        lines.append(tail)
        return "\n".join(lines)

    lines = [HEADER_COORD.format(w=maze.width, h=maze.height)]
    lines.append(coord)
    lines.append(_coords_block(maze) if style == "coords" else _rows_block(maze))
    lines.append(tail)
    return "\n".join(lines)


def render_observation(
    maze: Maze,
    pos: tuple[int, int],
    last_actions: list[str] | tuple[str, ...] = (),
    *,
    style: str = "minimal",
    visited: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
) -> str:
    """完整观测文本（= 状态块）。rollout 侧再套 chat template。"""
    return render_state(maze, pos, last_actions, style=style, visited=visited)
