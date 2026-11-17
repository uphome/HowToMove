from __future__ import annotations

import json

from agent import AgentGroup, MazeAgent
from env.maze import Maze, bfs_field
from rollout import Step, format_tool_call


class BatchPolicy:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = []

    def act_batch(self, messages_batch):
        self.calls.append(len(messages_batch))
        actions = next(self.batches)
        assert len(actions) == len(messages_batch)
        return [
            Step(text=format_tool_call(a) if a else "invalid", action=a)
            for a in actions
        ]


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.seen = []

    def act(self, messages):
        self.seen.append(list(messages))
        direction = next(self.actions)
        return Step(text=format_tool_call(direction), action=direction)


def make_maze(goal=(2, 0), walls=frozenset(), max_steps=5):
    dist = bfs_field(5, 5, walls, goal)
    return Maze(
        width=5, height=5, walls=walls, start=(0, 0), goal=goal,
        max_steps=max_steps, level="L0", seed=0, dist=dist,
    )


def test_agent_system_contains_static_map():
    run = MazeAgent(
        make_maze(), ScriptedPolicy(["right", "right"]),
        history_window=4,
    ).run()
    system = run.messages[0]["content"]
    assert "Start: (0, 0)" in system
    assert "Goal: (2, 0)" in system
    assert "Walls: none" in system
    assert run.messages[1]["content"] == "Begin the maze."


def test_agent_records_assistant_tool_history():
    policy = ScriptedPolicy(["right", "right"])
    run = MazeAgent(
        make_maze(), policy,
        history_window=4,
    ).run()
    assert run.success
    assert run.steps == 2
    assert [m["role"] for m in run.messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool"
    ]
    assert run.messages[2]["role"] == "assistant"
    assert '"direction": "right"' in run.messages[2]["content"]
    tool = json.loads(run.messages[3]["content"])
    assert tool["moved"] == "right"
    assert tool["pos"] == [1, 0]
    assert tool["blocked"] is False
    assert tool["reason"] == "You moved right."
    assert run.turns[0]["action"] == "right"
    assert run.turns[1]["pos"] == [2, 0]
    assert run.turns[-1]["tool_result"]["reason"] == "You reached the goal."
    assert run.turns[-1]["tool_result"]["done"] is True
    assert len(policy.seen) == 2
    assert policy.seen[1][-2]["role"] == "assistant"
    assert policy.seen[1][-1]["role"] == "tool"


def test_agent_wall_collision_keeps_position_and_explains_reason():
    run = MazeAgent(
        make_maze(goal=(2, 0), walls=frozenset({(1, 0)}), max_steps=1),
        ScriptedPolicy(["right"]),
        history_window=4,
    ).run()
    assert not run.success
    assert run.turns[0]["pos"] == [0, 0]
    result = run.turns[0]["tool_result"]
    assert result["blocked"] is True
    assert result["blocked_by"] == "wall"
    assert result["pos"] == [0, 0]
    assert result["reason"] == (
        "You hit a wall. Your position did not change."
    )


def test_agent_history_window_keeps_recent_pairs():
    policy = ScriptedPolicy(["right", "right", "right"])
    run = MazeAgent(
        make_maze(goal=(3, 0)), policy,
        history_window=1,
    ).run()
    assert run.success
    assert run.steps == 3
    assert [m["role"] for m in run.messages] == [
        "system", "user", "assistant", "tool"
    ]
    assert len(policy.seen[-1]) == 4


def test_agent_group_batches_active_trajectories():
    # g0: right,right -> goal
    # g1: invalid -> finishes first
    policy = BatchPolicy([
        ["right", None],
        ["right"],
    ])
    group = AgentGroup(
        make_maze(goal=(2, 0)),
        policy,
        g=2,
        history_window=4,
        episode_prefix="u1/m0",
    )
    runs = group.run()
    assert [r.episode_id for r in runs] == ["u1/m0/g0", "u1/m0/g1"]
    assert runs[0].success
    assert not runs[1].success
    assert policy.calls == [2, 1]
    assert runs[0].steps == 2
    assert runs[1].steps == 1
    assert runs[0].messages[2]["role"] == "assistant"
    assert runs[0].messages[3]["role"] == "tool"
    assert runs[1].turns[0]["tool_result"]["reason"] == (
        "Invalid tool call. The episode ended."
    )
