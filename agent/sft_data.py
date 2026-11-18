"""Build SFT turn samples for the static-map tool-calling agent."""

from __future__ import annotations

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


def build_agent_sft_items(
    records,
    tokenizer,
    *,
    levels: Iterable[str] = ("L0",),
    maps_per_level: int = 100,
    history_window: int = 12,
    max_seq_len: int = 4096,
    seed: int = 7,
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
    for rec in selected:
        maze = rec.to_maze()
        task_seed = (int(rec.hash[:16], 16) ^ (seed * 1000003)) & 0xFFFFFFFF
        path = random_shortest_path(maze, random.Random(task_seed))
        agent = MazeAgent(
            maze, None,
            history_window=history_window,
            episode_id=f"{rec.id}/sft",
        )
        for i in range(len(path) - 1):
            action = direction_between(path[i], path[i + 1])
            messages = agent.messages()
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
                continue
            items.append(AgentSFTItem(
                map_id=rec.id,
                level=rec.level,
                turn=i + 1,
                prompt_ids=prompt_ids,
                completion_ids=completion,
                action=action,
                prompt_text=prompt_text,
            ))
            actions[action] += 1
            # Apply the teacher action to obtain the exact next tool result.
            agent.apply(Step(
                text=format_tool_call(action),
                action=action,
                prompt_ids=prompt_ids,
                tokens=completion,
            ))
    return items, actions, len(selected)


__all__ = [
    "AgentSFTItem",
    "build_agent_sft_items",
    "direction_between",
    "random_shortest_path",
]
