from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
import zlib

from .maze import LEVELS, generate
from .viz import Canvas, build_report, render_ansi, render_png, render_sheet, render_svg


def test_canvas_png_roundtrip():
    c = Canvas(4, 3, bg=(10, 20, 30))
    c.rect(1, 1, 3, 2, (200, 100, 50))
    data = c.png_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (4, 3)

    # 解析 IDAT 回来校验像素
    pos = 8
    idat = b""
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        if tag == b"IDAT":
            idat += payload
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = 4 * 3 + 1
    assert len(raw) == stride * 3
    assert raw[0] == 0
    assert raw[1:4] == bytes((10, 20, 30))
    px = 1 * stride + 1 + 1 * 3
    assert raw[px : px + 3] == bytes((200, 100, 50))


def test_render_png(tmp_path):
    maze = generate(level="L3", seed=11)
    out = render_png(maze, tmp_path / "m.png", path=maze.shortest_path())
    assert out.exists() and out.stat().st_size > 1000


def test_render_sheet(tmp_path):
    entries = [(lv, generate(level=lv, seed=i)) for i, lv in enumerate(LEVELS)]
    out = render_sheet(entries, tmp_path / "sheet.png")
    assert out.exists() and out.stat().st_size > 5000


def test_render_svg_static():
    maze = generate(level="L1", seed=3)
    svg = render_svg(maze, path=maze.shortest_path(), title="t & <x>")
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "&amp;" in svg and "&lt;" in svg
    tags = [e.tag.split("}")[-1] for e in root.iter()]
    assert tags.count("polyline") == 1
    assert tags.count("circle") == 1
    assert tags.count("animate") == 0


def test_render_svg_animated():
    maze = generate(level="L2", seed=4)
    frames = maze.shortest_path()
    svg = render_svg(maze, path=frames, frames=frames)
    root = ET.fromstring(svg)
    tags = [e.tag.split("}")[-1] for e in root.iter()]
    assert tags.count("animate") == 2


def test_render_ansi_shape():
    maze = generate(level="L0", seed=0)
    text = render_ansi(maze, maze.start, color=False)
    lines = text.splitlines()
    assert len(lines) == maze.height
    assert all(len(line.split()) == maze.width for line in lines)
    assert text.count("A") >= 1 and text.count("G") == 1


def test_build_report(tmp_path):
    files = build_report(tmp_path, seed=11)
    names = {f.name for f in files}
    for lv in LEVELS:
        assert f"{lv}.png" in names
    assert {"levels.png", "L3_rollout.svg", "index.html"} <= names
    assert (tmp_path / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
