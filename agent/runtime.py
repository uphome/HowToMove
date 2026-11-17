"""A tool-calling maze agent with the static map in the system prompt.

The system prompt contains the complete static map:

    start, goal, walls, coordinate convention, and move-tool contract.

Each turn is a compact exchange:

    user: Begin the maze.
    assistant: <tool_call>move(...)</tool_call>
    tool: {"moved": ..., "pos": ..., "blocked": ..., "reason": ...}
    ...

The tool result deliberately does not repeat the full map or observation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from env.maze import ACTIONS, Maze, MazeEnv


def build_agent_system(maze: Maze) -> str:
    walls = " ".join(
        f"({x},{y})" for x, y in sorted(maze.walls, key=lambda p: (p[1], p[0]))
    ) or "none"
    return (
        f"You are an agent solving a {maze.width}x{maze.height} maze.\n"
        "Coordinates: x increases to the right, y increases downward.\n"
        f"Start: {maze.start}\n"
        f"Goal: {maze.goal}\n"
        f"Walls: {walls}\n\n"
        "You control one cell. Each turn, call the move tool with exactly one "
        "direction: up, down, left, or right.\n\n"
        "The move tool returns a compact JSON result:\n"
        "- moved: the direction you attempted\n"
        "- ok: whether the move succeeded\n"
        "- pos: your position after the move\n"
        "- blocked: whether the move was blocked\n"
        "- blocked_by: wall, boundary, or null\n"
        "- done: whether the episode ended\n"
        "- goal_delta: goal - pos\n"
        "- reason: a short explanation\n\n"
        "If you hit a wall or the map boundary, your position does not change.\n"
        "If done=true, stop calling move.\n"
        "Always call move. Never reply in plain text."
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
                                 if t["action"] is None),
            "collisions": sum(1 for t in self.turns
                              if t["tool_result"]["blocked"]),
        }


class MazeAgent:
    def __init__(
        self,
        maze: Maze,
        policy,
        *,
        history_window: int = 12,
        memory: int = 3,
        revisit_penalty: float = 0.0,
        system: str | None = None,
        on_step: Callable[[dict], None] | None = None,
    ):
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        self.maze = maze
        self.policy = policy
        self.history_window = history_window
        self.on_step = on_step
        self.env = MazeEnv(maze, memory=memory,
                           revisit_penalty=revisit_penalty)
        self.path: list[tuple[int, int]] = [maze.start]
        self.history: list[dict] = []
        self.turns: list[dict] = []
        self.total_reward = 0.0
        self.base_messages = [
            {"role": "system",
             "content": system or build_agent_system(maze)},
            {"role": "user", "content": "Begin the maze."},
        ]

    def messages(self) -> list[dict]:
        return self.base_messages + self.history

    def _trim(self) -> None:
        while len(self.history) > 2 * self.history_window:
            del self.history[:2]

    def _blocked_by(self, action: str | None) -> str | None:
        if action not in ACTIONS:
            return None
        dx, dy = ACTIONS[action]
        nx, ny = self.env.pos[0] + dx, self.env.pos[1] + dy
        if not (0 <= nx < self.maze.width and 0 <= ny < self.maze.height):
            return "boundary"
        if (nx, ny) in self.maze.walls:
            return "wall"
        return None

    def run(self) -> AgentRun:
        while not self.env.done:
            prompt_messages = self.messages()
            step = self.policy.act(prompt_messages)
            self.history.append({"role": "assistant", "content": step.text})

            action = step.action if step.action in ACTIONS else None
            before = self.env.pos
            blocked_by = self._blocked_by(action)
            result = self.env.step(action or "invalid")
            self.path.append(result.pos)
            self.total_reward += result.reward

            reason = result.info.get("reason", "unknown")
            if action is None:
                reason_text = "Invalid tool call. The episode ended."
            elif blocked_by == "wall":
                reason_text = ("You hit a wall. "
                               "Your position did not change.")
            elif blocked_by == "boundary":
                reason_text = ("You hit the map boundary. "
                               "Your position did not change.")
            elif reason == "goal":
                reason_text = "You reached the goal."
            elif reason == "truncated":
                reason_text = "The step limit was reached."
            else:
                reason_text = f"You moved {action}."

            payload = {
                "moved": action,
                "ok": action is not None and blocked_by is None,
                "pos": list(result.pos),
                "blocked": blocked_by is not None,
                "blocked_by": blocked_by,
                "done": bool(result.done),
                "goal_delta": [
                    self.maze.goal[0] - result.pos[0],
                    self.maze.goal[1] - result.pos[1],
                ],
                "reason": reason_text,
            }
            self.history.append({
                "role": "tool",
                "name": "move",
                "content": json.dumps(payload, ensure_ascii=False),
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


__all__ = ["AgentRun", "MazeAgent", "build_agent_system"]
