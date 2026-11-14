"""迷宫可视化：纯标准库（zlib + struct），不依赖 PIL / matplotlib。

产物
    PNG   静态图（单张或 5 个级别拼图）
    SVG   带标题的矢量图；给定 frames 时是逐格动画
    ANSI  终端 24-bit 彩色渲染

CLI
    python -m env.viz                 # 生成 out/ 下全部产物 + index.html
    python -m env.viz --out out2
"""

from __future__ import annotations

import argparse
import html
import struct
import zlib
from pathlib import Path

from .maze import AGENT, FREE, GOAL, LEVELS, WALL, Maze, manhattan

FREE_RGB = (248, 249, 252)
WALL_RGB = (52, 58, 70)
GRID_RGB = (226, 229, 236)
START_RGB = (86, 148, 240)
GOAL_RGB = (56, 190, 120)
AGENT_RGB = (250, 138, 40)
PATH_RGB = (168, 140, 228)

LEVEL_COLOR = {
    "L0": (86, 148, 240),
    "L1": (56, 190, 120),
    "L2": (245, 190, 60),
    "L3": (240, 100, 100),
    "L4": (168, 120, 230),
}


# ---------------------------------------------------------------- PNG


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = FREE_RGB):
        self.width = width
        self.height = height
        self.buf = bytearray(bytes(bg) * (width * height))

    def rect(self, x0: float, y0: float, x1: float, y1: float, rgb) -> None:
        x0 = max(0, int(round(x0)))
        y0 = max(0, int(round(y0)))
        x1 = min(self.width, int(round(x1)))
        y1 = min(self.height, int(round(y1)))
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(rgb) * (x1 - x0)
        for y in range(y0, y1):
            i = (y * self.width + x0) * 3
            self.buf[i : i + len(row)] = row

    def png_bytes(self) -> bytes:
        stride = self.width * 3
        raw = b"".join(
            b"\x00" + bytes(self.buf[y * stride : (y + 1) * stride])
            for y in range(self.height)
        )

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b"")
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.png_bytes())


def _center(cell: int, pos: tuple[int, int], ox: int = 0, oy: int = 0):
    return (ox + pos[0] * cell + cell / 2, oy + pos[1] * cell + cell / 2)


def draw_maze(
    canvas: Canvas,
    maze: Maze,
    *,
    cell: int = 36,
    ox: int = 0,
    oy: int = 0,
    path: list[tuple[int, int]] | None = None,
    pos: tuple[int, int] | None = None,
    show_path: bool = True,
) -> None:
    # 底色 / 墙
    for y in range(maze.height):
        for x in range(maze.width):
            rgb = WALL_RGB if (x, y) in maze.walls else FREE_RGB
            canvas.rect(ox + x * cell, oy + y * cell,
                        ox + (x + 1) * cell, oy + (y + 1) * cell, rgb)

    # 网格线
    for x in range(maze.width + 1):
        canvas.rect(ox + x * cell, oy, ox + x * cell + 1, oy + maze.height * cell, GRID_RGB)
    for y in range(maze.height + 1):
        canvas.rect(ox, oy + y * cell, ox + maze.width * cell, oy + y * cell + 1, GRID_RGB)

    # 最短路径
    if show_path and path:
        t = max(3, int(cell * 0.18))
        for a, b in zip(path, path[1:]):
            ax, ay = _center(cell, a, ox, oy)
            bx, by = _center(cell, b, ox, oy)
            if ay == by:
                canvas.rect(min(ax, bx) - t / 2, ay - t / 2, max(ax, bx) + t / 2, ay + t / 2, PATH_RGB)
            else:
                canvas.rect(ax - t / 2, min(ay, by) - t / 2, ax + t / 2, max(ay, by) + t / 2, PATH_RGB)
        for p in path:
            cx, cy = _center(cell, p, ox, oy)
            canvas.rect(cx - t / 2, cy - t / 2, cx + t / 2, cy + t / 2, PATH_RGB)

    # 起点 / 终点 / 当前位置
    inset = cell * 0.24
    for point, rgb in ((maze.start, START_RGB), (maze.goal, GOAL_RGB)):
        cx, cy = _center(cell, point, ox, oy)
        canvas.rect(cx - cell / 2 + inset, cy - cell / 2 + inset,
                    cx + cell / 2 - inset, cy + cell / 2 - inset, rgb)

    if pos is not None:
        cx, cy = _center(cell, pos, ox, oy)
        r = cell * 0.3
        canvas.rect(cx - r, cy - r, cx + r, cy + r, AGENT_RGB)


def render_png(
    maze: Maze,
    out: str | Path,
    *,
    cell: int = 40,
    path: list[tuple[int, int]] | None = None,
    pos: tuple[int, int] | None = None,
    pad: int = 12,
) -> Path:
    canvas = Canvas(maze.width * cell + 2 * pad, maze.height * cell + 2 * pad)
    draw_maze(canvas, maze, cell=cell, ox=pad, oy=pad, path=path, pos=pos)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def render_sheet(
    entries: list[tuple[str, Maze]],
    out: str | Path,
    *,
    cell: int = 30,
    gap: int = 14,
    cols: int | None = None,
) -> Path:
    cols = cols or len(entries)
    rows = (len(entries) + cols - 1) // cols
    mh = max(m.height for _, m in entries)
    mw = max(m.width for _, m in entries)
    canvas = Canvas(cols * mw * cell + (cols + 1) * gap,
                    rows * mh * cell + (rows + 1) * gap,
                    bg=(255, 255, 255))
    for i, (_, maze) in enumerate(entries):
        r, c = divmod(i, cols)
        draw_maze(
            canvas,
            maze,
            cell=cell,
            ox=gap + c * (mw * cell + gap),
            oy=gap + r * (mh * cell + gap),
            path=maze.shortest_path(),
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ---------------------------------------------------------------- SVG


def render_svg(
    maze: Maze,
    *,
    cell: int = 40,
    path: list[tuple[int, int]] | None = None,
    frames: list[tuple[int, int]] | None = None,
    title: str = "",
    duration: float | None = None,
) -> str:
    w = maze.width * cell
    h = maze.height * cell
    head = 46 if title else 0
    esc = html.escape

    def rect(x, y, ww, hh, fill, extra=""):
        return (f'<rect x="{x * cell}" y="{head + y * cell}" '
                f'width="{ww * cell}" height="{hh * cell}" fill="{fill}"{extra}/>')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + head}" '
        f'viewBox="0 0 {w} {h + head}" font-family="ui-monospace,Consolas,monospace">',
        f'<rect width="{w}" height="{h + head}" fill="#ffffff"/>',
    ]
    if title:
        parts.append(
            f'<text x="0" y="28" font-size="18" fill="#20242c">{esc(title)}</text>'
        )

    parts.append(rect(0, 0, maze.width, maze.height, "#f8f9fc"))
    for y in range(maze.height):
        for x in range(maze.width):
            if (x, y) in maze.walls:
                parts.append(rect(x, y, 1, 1, "#343a46"))

    stroke = f'stroke="#e2e5ec" stroke-width="1"'
    for x in range(maze.width + 1):
        parts.append(f'<line x1="{x * cell}" y1="{head}" x2="{x * cell}" '
                     f'y2="{head + h}" {stroke}/>')
    for y in range(maze.height + 1):
        parts.append(f'<line x1="0" y1="{head + y * cell}" x2="{w}" '
                     f'y2="{head + y * cell}" {stroke}/>')

    if path:
        pts = " ".join(f"{x * cell + cell / 2},{head + y * cell + cell / 2}" for x, y in path)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#a88ce4" '
            f'stroke-width="{cell * 0.18:.1f}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )

    for (x, y), rgb in (
        (maze.start, "#5694f0"),
        (maze.goal, "#38be78"),
    ):
        parts.append(rect(x + 0.24, y + 0.24, 0.52, 0.52, rgb, ' rx="3"'))

    spot = frames[0] if frames else (maze.start if path is None else path[0])
    cx, cy = spot[0] * cell + cell / 2, head + spot[1] * cell + cell / 2
    r = cell * 0.3
    anim = ""
    if frames and len(frames) > 1:
        dur = duration or 0.45 * len(frames)
        values = ";".join(f"{x * cell + cell / 2}" for x, y in frames)
        values_y = ";".join(f"{head + y * cell + cell / 2}" for x, y in frames)
        times = ";".join(f"{i / (len(frames) - 1):.4f}" for i in range(len(frames)))
        anim = (
            f'<animate attributeName="cx" values="{values}" keyTimes="{times}" '
            f'dur="{dur:.1f}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="cy" values="{values_y}" keyTimes="{times}" '
            f'dur="{dur:.1f}s" repeatCount="indefinite" calcMode="discrete"/>'
        )
    parts.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="#fa8a28">{anim}</circle>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- ANSI


def render_ansi(
    maze: Maze,
    pos: tuple[int, int] | None = None,
    path: list[tuple[int, int]] | None = None,
    *,
    color: bool = True,
    cell: str = "  ",
) -> str:
    path_set = set(path or ())
    lines = []
    for y in range(maze.height):
        row = []
        for x in range(maze.width):
            if (x, y) in maze.walls:
                rgb, ch = WALL_RGB, WALL
            elif pos == (x, y):
                rgb, ch = AGENT_RGB, AGENT
            elif (x, y) == maze.goal:
                rgb, ch = GOAL_RGB, GOAL
            elif (x, y) == maze.start:
                rgb, ch = START_RGB, "S"
            elif (x, y) in path_set:
                rgb, ch = PATH_RGB, "*"
            else:
                rgb, ch = FREE_RGB, FREE
            if color:
                row.append(f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m{cell}\x1b[0m")
            else:
                row.append(f"{ch} ")
        lines.append("".join(row))
    return "\n".join(lines)


# ---------------------------------------------------------------- report


def maze_for(level: str, seed: int, max_steps: int = 30) -> Maze:
    from .maze import generate

    return generate(level=level, seed=seed, max_steps=max_steps)


def build_report(
    outdir: str | Path = "out",
    *,
    seed: int = 11,
    cell: int = 40,
) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    entries: list[tuple[str, Maze]] = []

    for i, level in enumerate(LEVELS):
        maze = maze_for(level, seed + i)
        entries.append((level, maze))
        written.append(
            render_png(maze, outdir / f"{level}.png", cell=cell,
                       path=maze.shortest_path(), pos=maze.start)
        )

    written.append(render_sheet(entries, outdir / "levels.png"))

    demo_level = "L3"
    demo = maze_for(demo_level, seed + 103)
    frames = demo.shortest_path()

    # 静态：只画路径
    written.append(
        render_png(demo, outdir / "L3_path.png", cell=cell, path=frames, pos=demo.start)
    )

    # 动画：agent 沿最短路逐格移动
    svg = render_svg(
        demo,
        cell=cell,
        path=frames,
        frames=frames,
        title=f"{demo_level}  seed={seed + 3}  shortest={demo.path_length}"
              f"  manhattan={manhattan(demo.start, demo.goal)}  detour={demo.detour}",
    )
    svg_path = outdir / "L3_rollout.svg"
    svg_path.write_text(svg, encoding="utf-8")
    written.append(svg_path)

    blocks = []
    for level, maze in entries:
        blocks.append(
            f'<figure><img src="{level}.png" alt="{level}">'
            f'<figcaption style="color:rgb{LEVEL_COLOR[level]}">'
            f'{level} &middot; shortest={maze.path_length} &middot; detour={maze.detour}'
            f' &middot; walls={len(maze.walls)}</figcaption></figure>'
        )
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>AgenticRL mazes</title>
<style>
  body {{ margin: 0; padding: 28px; background: #fff;
         font-family: ui-monospace, Consolas, monospace; color: #20242c; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 4px; }}
  p.meta {{ color: #6b7280; font-size: 13px; margin: 0 0 20px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 18px; margin-bottom: 28px; }}
  figure {{ margin: 0; }}
  img {{ display: block; border: 1px solid #e2e5ec; border-radius: 4px; }}
  figcaption {{ font-size: 12px; margin-top: 6px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 0 0 10px; }}
  .svg-wrap {{ border: 1px solid #e2e5ec; border-radius: 4px; display: inline-block; }}
</style>
<h1>AgenticRL &middot; 12&times;12 maze corpus</h1>
<p class="meta">detour = shortest &minus; manhattan（恒为偶数，4 表示至少走两步远离目标）</p>
<div class="grid">{''.join(blocks)}</div>
<h2>L3 rollout &middot; 逐格动画</h2>
<div class="svg-wrap">{svg}</div>
"""
    (outdir / "index.html").write_text(page, encoding="utf-8")
    written.append(outdir / "index.html")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="render 12x12 mazes")
    parser.add_argument("--out", default="out")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cell", type=int, default=40)
    parser.add_argument("--level", default=None, choices=LEVELS)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    if args.level:
        maze = maze_for(args.level, args.seed)
        print(render_ansi(maze, maze.start, maze.shortest_path(), color=not args.no_color))
        print(f"start={maze.start} goal={maze.goal} shortest={maze.path_length} "
              f"detour={maze.detour} walls={len(maze.walls)}")
        return

    for path in build_report(args.out, seed=args.seed, cell=args.cell):
        print(path)


if __name__ == "__main__":
    main()
