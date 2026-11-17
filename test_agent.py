from __future__ import annotations

from agent import MazeAgent
from env.maze import Maze, bfs_field
from rollout import Step, format_tool_call


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.seen = []

    def act(self, messages):
        self.seen.append(list(messages))
        direction = next(self.actions)
        return Step(text=format_tool_call(direction), action=direction)


def make_maze(goal=(2, 0), max_steps=5):
    walls = frozenset()
    dist = bfs_field(5, 5, walls, goal)
    return Maze(
        width=5, height=5, walls=walls, start=(0, 0), goal=goal,
        max_steps=max_steps, level="L0", seed=0, dist=dist,
    )


def test_agent_records_assistant_tool_history():
    policy = ScriptedPolicy(["right", "right"])
    run = MazeAgent(
        make_maze(), policy,
        obs_style="minimal", fewshot=None, history_window=4,
        show_visited=False,
    ).run()
    assert run.success
    assert run.steps == 2
    assert [m["role"] for m in run.messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool"
    ]
    assert run.messages[2]["role"] == "assistant"
    assert '"direction": "right"' in run.messages[2]["content"]
    assert '"moved": "right"' in run.messages[3]["content"]
    assert run.turns[0]["action"] == "right"
    assert run.turns[1]["pos"] == [2, 0]
    assert len(policy.seen) == 2
    assert policy.seen[1][-2]["role"] == "assistant"
    assert policy.seen[1][-1]["role"] == "tool"


def test_agent_history_window_keeps_recent_pairs():
    policy = ScriptedPolicy(["right", "right", "right"])
    run = MazeAgent(
        make_maze(goal=(3, 0)), policy,
        obs_style="minimal", fewshot=None, history_window=1,
        show_visited=False,
    ).run()
    assert run.success
    assert run.steps == 3
    assert [m["role"] for m in run.messages] == [
        "system", "user", "assistant", "tool"
    ]
    assert len(policy.seen[-1]) == 4
