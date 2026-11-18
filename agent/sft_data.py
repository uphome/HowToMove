"""Build SFT turn samples for the static-map tool-calling agent."""

from __future__ import annotations

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


def build_agent_sft_items(
    records,
    tokenizer,
    *,
    levels: Iterable[str] = ("L0",),
    maps_per_level: int = 100,
    history_window: int = 12,
    max_seq_len: int = 4096,
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
        path = maze.shortest_path()
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


__all__ = ["AgentSFTItem", "build_agent_sft_items", "direction_between"]
