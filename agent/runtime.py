"""A small tool-calling maze agent with explicit conversation history.

Each turn is a real assistant/tool exchange:

    system
    user: initial observation
    assistant: <tool_call>move(...)</tool_call>
    tool: {"moved": ..., "reward": ..., "observation": ...}
    assistant: <tool_call>move(...)</tool_call>
    ...

The environment result is returned through the `tool` role, and the next
assistant call sees the full recent conversation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from env.maze import ACTIONS, Maze, MazeEnv
from env.render import render_observation
from prompts import REL_FEWSHOT, REL_FEWSHOT_L1, REL_SYSTEM


AGENT_SYSTEM = (
    REL_SYSTEM
    + " After every move, the move tool returns a JSON tool result containing "
      "the new observation, reward, done, and reason. Use that result to decide "
      "the next move."
)


@dataclass
class AgentRun:
    maze: Maze
    messages: list[dict]
    turns: list[dict] = field(default_factory=list)
    success: bool = False
    truncated: bool = False
    total_reward: float = 0.0

    @property
    def steps(self) -> int:
        return len(self.turns)

    def summary(self) -> dict:
        return {
            "level": self.maze.level,
            "seed": self.maze.seed,
            "steps": self.steps,
            "success": self.success,
            "truncated": self.truncated,
            "total_reward": round(self.total_reward, 3),
            "shortest": self.maze.path_length,
            "format_errors": sum(1 for t in self.turns
                                 if t["action"] == "invalid"),
            "collisions": sum(1 for t in self.turns
                              if t["info"].get("reason") == "collision"),
        }


def _fewshot_messages(kind: str | None) -> list[dict]:
    if not kind or kind == "none":
        return []
    if kind == "rel":
        examples = REL_FEWSHOT
    elif kind == "rel_l1":
        examples = REL_FEWSHOT_L1
    else:
        raise ValueError(f"unknown fewshot kind: {kind}")
    out = []
    for state, direction in examples:
        out.append({"role": "user", "content": state})
        out.append({
            "role": "assistant",
            "content": (
                '<tool_call>\n'
                '{"name": "move", "arguments": {"direction": "%s"}}\n'
                '</tool_call>' % direction
            ),
        })
    return out


class MazeAgent:
    def __init__(
        self,
        maze: Maze,
        policy,
        *,
        obs_style: str = "rel",
        fewshot: str | None = "rel_l1",
        history_window: int = 12,
        memory: int = 3,
        show_visited: bool = True,
        revisit_penalty: float = 0.0,
        system: str = AGENT_SYSTEM,
        on_step: Callable[[dict], None] | None = None,
    ):
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        self.maze = maze
        self.policy = policy
        self.obs_style = obs_style
        self.history_window = history_window
        self.memory = memory
        self.show_visited = show_visited
        self.revisit_penalty = revisit_penalty
        self.on_step = on_step
        self.env = MazeEnv(maze, memory=memory,
                           revisit_penalty=revisit_penalty)
        self.path: list[tuple[int, int]] = [maze.start]
        self.history: list[dict] = []
        self.turns: list[dict] = []
        self.total_reward = 0.0
        self.base_messages = (
            [{"role": "system", "content": system}]
            + _fewshot_messages(fewshot)
            + [{"role": "user", "content": self._observation()}]
        )

    def _observation(self) -> str:
        visited = tuple(self.path[:-1]) if self.show_visited else ()
        return render_observation(
            self.maze, self.env.pos, self.env.last_actions,
            style=self.obs_style, visited=visited,
        )

    def messages(self) -> list[dict]:
        return self.base_messages + self.history

    def _trim(self) -> None:
        # history is assistant/tool pairs
        while len(self.history) > 2 * self.history_window:
            del self.history[:2]

    def run(self) -> AgentRun:
        while not self.env.done:
            prompt_messages = self.messages()
            step = self.policy.act(prompt_messages)
            self.history.append({"role": "assistant", "content": step.text})

            action = step.action if step.action in ACTIONS else "invalid"
            before = self.env.pos
            result = self.env.step(action)
            self.path.append(result.pos)
            self.total_reward += result.reward
            observation = self._observation()
            reason = result.info.get("reason", "unknown")
            payload = {
                "ok": action in ACTIONS and reason != "collision",
                "moved": action,
                "from": list(before),
                "pos": list(result.pos),
                "reward": round(result.reward, 4),
                "done": bool(result.done),
                "reason": reason,
                "observation": observation,
            }
            tool_text = json.dumps(payload, ensure_ascii=False)
            self.history.append({
                "role": "tool",
                "name": "move",
                "content": tool_text,
            })
            self._trim()

            turn = {
                "turn": len(self.turns) + 1,
                "pos": list(result.pos),
                "action": action,
                "reward": round(result.reward, 4),
                "done": bool(result.done),
                "info": dict(result.info),
                "tool_result": payload,
                "assistant_text": step.text,
                "prompt": prompt_messages,
                "logprob": step.logprob,
            }
            self.turns.append(turn)
            if self.on_step is not None:
                self.on_step(turn)

        success = self.env.terminated and self.env.pos == self.maze.goal
        return AgentRun(
            maze=self.maze,
            messages=self.messages(),
            turns=self.turns,
            success=success,
            truncated=self.env.truncated,
            total_reward=self.total_reward,
        )


__all__ = ["AGENT_SYSTEM", "AgentRun", "MazeAgent"]
