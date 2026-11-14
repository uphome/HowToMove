from __future__ import annotations

from env.maze import ACTIONS, generate
from env.ui import UIServer
from rollout import (
    MockPolicy,
    Observation,
    Step,
    format_tool_call,
    parse_action,
    run_episode,
)


class OraclePolicy:
    """沿 BFS 距离场下降，用来验证采集链路本身是对的。"""

    def act(self, obs: Observation) -> Step:
        for name, (dx, dy) in ACTIONS.items():
            nxt = (obs.pos[0] + dx, obs.pos[1] + dy)
            if obs.maze.is_free(*nxt) and obs.maze.distance(nxt) == obs.maze.distance(obs.pos) - 1:
                text = format_tool_call(name)
                return Step(text=text, action=name, tokens=[1] * 12,
                            logprob=-0.01, latency=0.0)
        raise AssertionError(f"stuck at {obs.pos}")


class FormatErrorPolicy:
    def act(self, obs: Observation) -> Step:
        return Step(text="I cannot help with that.", action=None)


def test_format_tool_call_roundtrip():
    for name in ACTIONS:
        assert parse_action(format_tool_call(name)) == name


def test_parse_action_rejects_garbage():
    assert parse_action("") is None
    assert parse_action("move up please") is None
    assert parse_action('<tool_call>{"name":"move"}</tool_call>') is None
    assert parse_action('<tool_call>{"name":"move","arguments":{"direction":"northeast"}}'
                        "</tool_call>") is None
    assert parse_action('<tool_call>{not json}</tool_call>') is None


def test_parse_action_accepts_flat_shape():
    text = '<tool_call>{"action": "up"}</tool_call>'
    assert parse_action(text) == "up"


def test_oracle_episode_succeeds():
    maze = generate(level="L3", seed=114)
    traj = run_episode(maze, OraclePolicy())
    assert traj.success and not traj.truncated
    assert traj.steps == maze.path_length
    assert traj.total_reward > 40
    assert traj.summary()["format_errors"] == 0
    assert traj.path[-1] == maze.goal
    assert len(traj.path) == maze.path_length + 1


def test_format_error_terminates_episode():
    maze = generate(level="L0", seed=1)
    traj = run_episode(maze, FormatErrorPolicy())
    assert traj.steps == 1
    assert traj.turns[0].action == "invalid"
    assert traj.turns[0].reward == -10.1
    assert not traj.success


def test_mock_policy_runs_all_levels():
    for level in ("L0", "L1", "L3"):
        maze = generate(level=level, seed=3)
        traj = run_episode(maze, MockPolicy(seed=3))
        assert 1 <= traj.steps <= maze.max_steps
        assert all(t.text for t in traj.turns)


def test_ui_receives_llm_and_prompt():
    ui = UIServer(port=0).start()
    try:
        maze = generate(level="L0", seed=0)
        traj = run_episode(maze, OraclePolicy(), ui=ui, episode="unit")
        frame = ui.hub.snapshot()
        assert frame["episode"] == "unit"
        assert frame["turn"] == traj.steps
        assert frame["llm"] is not None
        assert "<tool_call>" in frame["llm"]["text"]
        assert frame["llm"]["action"] in ACTIONS
        assert "pos=(" in frame["prompt"] and "walls:" in frame["prompt"]
        assert frame["mode"] == "live"
    finally:
        ui.stop()


def test_episode_observation_options():
    maze = generate(level="L0", seed=3)
    traj = run_episode(maze, OraclePolicy(), obs_style="rows",
                       memory=0, show_visited=True)
    assert traj.success
    assert all(t.prompt.startswith("maze 12x12") for t in traj.turns)
    assert "last=none" in traj.turns[0].prompt
