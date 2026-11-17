"""Coordinate G independent MazeAgents through one batched policy."""

from __future__ import annotations

from typing import Callable

from .runtime import AgentRun, MazeAgent


class AgentGroup:
    """G agent states, one shared ChatPolicy/engine.

    Each turn collects the messages of all unfinished trajectories and sends
    them through one `policy.act_batch(...)` call. There are no Python threads.
    """

    def __init__(
        self,
        maze,
        policy,
        *,
        g: int = 8,
        history_window: int = 12,
        memory: int = 3,
        revisit_penalty: float = 0.0,
        episode_prefix: str = "group",
        system: str | None = None,
        on_step: Callable[[int, dict], None] | None = None,
    ):
        if g < 1:
            raise ValueError("g must be >= 1")
        self.maze = maze
        self.policy = policy
        self.g = g
        self.on_step = on_step
        self.agents: list[MazeAgent] = []
        for i in range(g):
            def _on_step(turn: dict, index: int = i) -> None:
                if self.on_step is not None:
                    self.on_step(index, turn)

            self.agents.append(MazeAgent(
                maze,
                None,
                history_window=history_window,
                memory=memory,
                revisit_penalty=revisit_penalty,
                system=system,
                on_step=_on_step,
                episode_id=f"{episode_prefix}/g{i}",
            ))

    def run(self) -> list[AgentRun]:
        active = list(self.agents)
        while active:
            messages_batch = [agent.messages() for agent in active]
            batch_fn = getattr(self.policy, "act_batch", None)
            if batch_fn is None:
                steps = [self.policy.act(m) for m in messages_batch]
            else:
                steps = batch_fn(messages_batch)
            if len(steps) != len(active):
                raise RuntimeError(
                    f"policy returned {len(steps)} steps for "
                    f"{len(active)} active agents")
            for agent, step in zip(active, steps):
                agent.apply(step)
            active = [agent for agent in active if not agent.done]
        return [agent.to_run() for agent in self.agents]


__all__ = ["AgentGroup"]
