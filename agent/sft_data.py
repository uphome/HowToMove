"""Build SFT turn samples for the static-map tool-calling agent."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from env.maze import ACTIONS
from prompts import TOOLS
from rollout import Step, format_tool_call

from .runtime import MazeAgent


@dataclass
class AgentSFTItem:
    map_id: str
    level: str
    turn: int
    prompt_ids: list[int]
    completion_ids: list[int]
    action: str
    prompt_text: str


def direction_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    d = (b[0] - a[0], b[1] - a[1])
    for name, delta in ACTIONS.items():
        if delta == d:
            return name
    raise ValueError(f"non-adjacent path step {a} -> {b}")


def random_shortest_path(maze, rng: random.Random) -> list[tuple[int, int]]:
    """Pick uniformly among all legal distance-decreasing neighbours.

    The fixed BFS traversal in Maze.shortest_path() always tries up/down
    first, which biases the first action of SFT trajectories.
    """
    path = [maze.start]
    pos = maze.start
    d = maze.distance(pos)
    while pos != maze.goal:
        choices = []
        for name, (dx, dy) in ACTIONS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if maze.is_free(*nxt) and maze.distance(nxt) == d - 1:
                choices.append((name, nxt))
        if not choices:
            raise RuntimeError(f"no descending neighbour from {pos}")
        name, nxt = rng.choice(choices)
        path.append(nxt)
        pos = nxt
        d -= 1
    return path


def blocked_directions(maze, pos: tuple[int, int]) -> list[tuple[str, str]]:
    """Return directions blocked by a wall or the boundary at ``pos``."""
    blocked = []
    for name, (dx, dy) in ACTIONS.items():
        nxt = (pos[0] + dx, pos[1] + dy)
        if not (0 <= nxt[0] < maze.width and 0 <= nxt[1] < maze.height):
            blocked.append((name, "boundary"))
        elif nxt in maze.walls:
            blocked.append((name, "wall"))
    return blocked


def _blocked_payload(maze, pos: tuple[int, int], action: str,
                     blocked_by: str) -> dict:
    reason = (
        "You hit a wall. Your position did not change."
        if blocked_by == "wall"
        else "You hit the map boundary. Your position did not change."
    )
    return {
        "moved": action,
        "ok": False,
        "pos": list(pos),
        "blocked": True,
        "blocked_by": blocked_by,
        "done": False,
        "goal_delta": [
            maze.goal[0] - pos[0],
            maze.goal[1] - pos[1],
        ],
        "reason": reason,
    }


def _collision_messages(agent, action: str, payload: dict) -> list[dict]:
    """Build the prompt after a blocked move without mutating the agent."""
    history = list(agent.history)
    # Mirror MazeAgent._trim() after the synthetic assistant/tool pair.
    while len(history) + 2 > 2 * agent.history_window:
        del history[:2]
    return agent.base_messages + history + [
        {"role": "assistant", "content": format_tool_call(action)},
        {
            "role": "tool",
            "name": "move",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def build_agent_sft_items(
    records,
    tokenizer,
    *,
    levels: Iterable[str] = ("L0",),
    maps_per_level: int = 100,
    history_window: int = 12,
    max_seq_len: int = 4096,
    seed: int = 7,
    recovery_per_task: int = 0,
) -> tuple[list[AgentSFTItem], Counter, int]:
    levels = tuple(levels)
    by_level = {lv: [] for lv in levels}
    for rec in records:
        if rec.level in by_level:
            by_level[rec.level].append(rec)
    selected = []
    for lv in levels:
        pool = sorted(by_level[lv], key=lambda r: r.hash)
        selected.extend(pool[:maps_per_level])

    eos = tokenizer.eos_token_id
    items: list[AgentSFTItem] = []
    actions: Counter = Counter()

    def add_item(messages: list[dict], action: str, turn: int) -> None:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=TOOLS,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text)
        completion = tokenizer.encode(
            format_tool_call(action), add_special_tokens=False)
        if eos is not None and (not completion or completion[-1] != eos):
            completion.append(eos)
        if len(prompt_ids) + len(completion) > max_seq_len:
            return
        items.append(AgentSFTItem(
            map_id=rec.id,
            level=rec.level,
            turn=turn,
            prompt_ids=prompt_ids,
            completion_ids=completion,
            action=action,
            prompt_text=prompt_text,
        ))
        actions[action] += 1

    for rec in selected:
        maze = rec.to_maze()
        task_seed = (int(rec.hash[:16], 16) ^ (seed * 1000003)) & 0xFFFFFFFF
        rng = random.Random(task_seed)
        path = random_shortest_path(maze, rng)
        agent = MazeAgent(
            maze, None,
            history_window=history_window,
            episode_id=f"{rec.id}/sft",
        )
        candidates = [
            i for i in range(len(path) - 1)
            if blocked_directions(maze, path[i])
        ]
        recovery_at = set()
        if recovery_per_task > 0 and candidates:
            recovery_at = set(rng.sample(
                candidates, min(recovery_per_task, len(candidates))))

        for i in range(len(path) - 1):
            action = direction_between(path[i], path[i + 1])
            messages = agent.messages()
            add_item(messages, action, i + 1)

            if i in recovery_at:
                blocked_action, blocked_by = rng.choice(
                    blocked_directions(maze, path[i]))
                payload = _blocked_payload(
                    maze, path[i], blocked_action, blocked_by)
                add_item(
                    _collision_messages(agent, blocked_action, payload),
                    action,
                    i + 1,
                )

            # Apply the teacher action to obtain the exact next tool result.
            completion = tokenizer.encode(
                format_tool_call(action), add_special_tokens=False)
            agent.apply(Step(
                text=format_tool_call(action),
                action=action,
                prompt_ids=None,
                tokens=completion,
            ))
    return items, actions, len(selected)


__all__ = [
    "AgentSFTItem",
    "build_agent_sft_items",
    "direction_between",
    "random_shortest_path",
]
