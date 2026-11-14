from __future__ import annotations

import json
import urllib.request

import pytest

from .maze import generate
from .ui import UIServer, VizClient, build_frame


@pytest.fixture()
def ui():
    server = UIServer(port=0).start()
    server.set_maze("L2", seed=5)
    yield server
    server.stop()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read().decode("utf-8"))


def test_frame_schema():
    maze = generate(level="L3", seed=1)
    frame = build_frame(maze, maze.start, path=maze.shortest_path(),
                        info={"reward": 1.0})
    assert frame["width"] == 12 and frame["height"] == 12
    assert frame["start"] == list(maze.start)
    assert frame["goal"] == list(maze.goal)
    assert len(frame["walls"]) == len(maze.walls)
    assert frame["path"][0] == list(maze.start)
    assert frame["meta"]["shortest"] == maze.path_length
    json.dumps(frame)  # 必须可序列化


def test_page_and_health(ui):
    with urllib.request.urlopen(f"{ui.url}api/health", timeout=3) as r:
        assert json.loads(r.read())["ok"] is True
    with urllib.request.urlopen(ui.url, timeout=3) as r:
        html = r.read().decode("utf-8")
    assert "<canvas" in html and "EventSource" in html


def test_maze_endpoint(ui):
    out = _post(f"{ui.url}api/maze", {"level": "L4", "seed": 9})
    assert out["level"] == "L4" and out["seed"] == 9
    state = _get(f"{ui.url}api/state")
    assert state["level"] == "L4" and state["meta"]["walls"] > 0


def test_maze_endpoint_rejects_bad_level(ui):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{ui.url}api/maze", {"level": "L9"})
    assert exc.value.code == 400


def test_action_endpoint(ui):
    before = _get(f"{ui.url}api/state")
    start = tuple(before["pos"])
    maze_shortest = before["meta"]["shortest"]
    assert maze_shortest > 0
    out = _post(f"{ui.url}api/action", {"action": "up"})
    assert "pos" in out
    after = _get(f"{ui.url}api/state")
    assert after["info"]["action"] == "up"
    assert isinstance(after["info"]["reward"], (int, float))


def test_frame_endpoint(ui):
    maze = generate(level="L0", seed=2)
    frame = build_frame(maze, maze.start, info={"tag": "external"})
    _post(f"{ui.url}api/frame", frame)
    state = _get(f"{ui.url}api/state")
    assert state["info"]["tag"] == "external"
    assert state["level"] == "L0"


def test_sse_first_event(ui):
    with urllib.request.urlopen(f"{ui.url}api/stream", timeout=3) as r:
        assert r.headers["Content-Type"].startswith("text/event-stream")
        line = r.readline().decode("utf-8")
    assert line.startswith("data: ")
    frame = json.loads(line[len("data: "):])
    assert frame["width"] == 12


def test_viz_client(ui):
    client = VizClient(ui.url)
    assert client.maze("L1", 3)
    assert client.action("right")
    maze = generate(level="L4", seed=4)
    assert client.push(build_frame(maze, maze.start))
    assert _get(f"{ui.url}api/state")["level"] == "L4"


def test_viz_client_failure_is_silent():
    client = VizClient("http://127.0.0.1:9", timeout=0.05)
    assert client.push({"x": 1}) is False
    assert client.failures == 1
