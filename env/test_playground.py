from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from .playground import PlaygroundRunner, parse_playground_map
from .ui import UIServer


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


@pytest.fixture()
def ui():
    server = UIServer(port=0).start()
    yield server
    server.stop()


def test_parse_custom_map():
    maze = parse_playground_map({
        "width": 6,
        "height": 5,
        "walls": [[2, 1], [2, 2], [2, 3]],
        "start": [0, 0],
        "goal": [5, 4],
        "max_steps": 30,
    })
    assert (maze.width, maze.height) == (6, 5)
    assert maze.path_length == 9
    assert maze.level == "CUSTOM"


def test_reject_start_on_wall():
    with pytest.raises(ValueError, match="start cannot be a wall"):
        parse_playground_map({
            "width": 5, "height": 5,
            "walls": [[0, 0]], "start": [0, 0], "goal": [4, 4],
        })


def test_reject_unreachable_goal():
    with pytest.raises(ValueError, match="unreachable"):
        parse_playground_map({
            "width": 5, "height": 5,
            "walls": [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]],
            "start": [0, 0], "goal": [4, 4], "max_steps": 30,
        })


def test_reject_short_budget():
    with pytest.raises(ValueError, match="shorter than the shortest path"):
        parse_playground_map({
            "width": 5, "height": 5, "walls": [],
            "start": [0, 0], "goal": [4, 4], "max_steps": 3,
        })


def test_runner_defaults_and_status():
    runner = PlaygroundRunner()
    cfg = runner.config()
    assert cfg["limits"]["min_size"] == 4
    assert cfg["limits"]["max_size"] == 40
    status = runner.status()
    assert status["state"] == "idle"
    assert status["model_loaded"] is False


def test_playground_page_and_config(ui):
    with urllib.request.urlopen(ui.url + "playground", timeout=3) as resp:
        html = resp.read().decode("utf-8")
    assert "Maze Playground" in html
    assert "画墙" in html
    assert "/api/playground/run" in html
    cfg = _get(ui.url + "api/playground/config")
    assert cfg["limits"]["max_size"] == 40
    assert "min_size" in cfg["limits"]


def test_custom_maze_endpoint(ui):
    out = _post(ui.url + "api/playground/maze", {
        "width": 7, "height": 6, "walls": [[3, 1], [3, 2]],
        "start": [0, 0], "goal": [6, 5], "max_steps": 30,
    })
    assert out["ok"] is True
    state = _get(ui.url + "api/state")
    assert (state["width"], state["height"]) == (7, 6)
    assert state["mode"] == "playground_reset"
    assert state["meta"]["shortest"] > 0


def test_run_requires_custom_maze(ui):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(ui.url + "api/playground/run", {})
    assert exc.value.code == 400


def test_custom_maze_rejects_invalid(ui):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(ui.url + "api/playground/maze", {
            "width": 5, "height": 5, "walls": [],
            "start": [0, 0], "goal": [0, 0], "max_steps": 10,
        })
    assert exc.value.code == 400
