"""轨迹采集：Policy → 多轮 while → Trajectory。

这一层现在是 mock，但接口就是最终形态：
    M3 只需要把 MockPolicy 换成 EnginePolicy（fork 出来的推理引擎），
    run_episode / UI 推送 / 轨迹记录 一行都不用改。

    EnginePolicy.act(obs) 内部：
        tokens, logprobs = engine.generate_batch([obs.prompt], ...)
        text = tokenizer.decode(tokens)
        return Step(text, parse_action(text), tokens, logprobs, latency)
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from env.maze import ACTIONS, Maze, MazeEnv, manhattan
from env.render import render_observation

if TYPE_CHECKING:  # pragma: no cover
    from env.ui import UIServer

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


# ---------------------------------------------------------------- 数据结构


@dataclass
class Observation:
    """模型能看到的一切。真实训练时只有 prompt 是输入，其余字段供 mock / 调试用。"""

    prompt: str
    maze: Maze
    pos: tuple[int, int]
    last_actions: list[str]
    turn: int


@dataclass
class Step:
    text: str
    action: str | None
    tokens: list[int] | None = None
    logprob: float | None = None
    latency: float = 0.0
    finish: str = "stop"
    prompt_ids: list[int] | None = None
    token_logprobs: list[float] | None = None


@dataclass
class Turn:
    turn: int
    prompt: str
    text: str
    action: str
    reward: float
    done: bool
    info: dict[str, Any]
    logprob: float | None = None
    pos: tuple[int, int] = (0, 0)
    prompt_ids: list[int] | None = None
    tokens: list[int] | None = None
    token_logprobs: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "prompt": self.prompt,
            "text": self.text,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
            "logprob": self.logprob,
            "pos": list(self.pos),
        }


@dataclass
class Trajectory:
    maze: Maze
    turns: list[Turn] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    truncated: bool = False

    @property
    def steps(self) -> int:
        return len(self.turns)

    @property
    def path(self) -> list[tuple[int, int]]:
        return [self.maze.start] + [t.pos for t in self.turns]

    def summary(self) -> dict[str, Any]:
        return {
            "level": self.maze.level,
            "seed": self.maze.seed,
            "steps": self.steps,
            "success": self.success,
            "truncated": self.truncated,
            "total_reward": round(self.total_reward, 3),
            "shortest": self.maze.path_length,
            "format_errors": sum(1 for t in self.turns if t.action == "invalid"),
            "collisions": sum(1 for t in self.turns if t.info.get("reason") == "collision"),
        }


# ---------------------------------------------------------------- 解析


def format_tool_call(direction: str) -> str:
    return ('<tool_call>\n'
            '{"name": "move", "arguments": {"direction": "%s"}}\n'
            '</tool_call>' % direction)


def parse_action(text: str) -> str | None:
    """从模型输出里抽 direction；不合法返回 None（触发格式惩罚）。"""
    match = TOOL_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    args = obj.get("arguments")
    if isinstance(args, dict):
        direction = args.get("direction")
    else:
        direction = obj.get("direction") or obj.get("action")
    return direction if direction in ACTIONS else None


# ---------------------------------------------------------------- Policy


class Policy(Protocol):
    def act(self, obs: Observation) -> Step: ...


class MockPolicy:
    """模拟一个"只学会了朝目标走"的 0.6B。

    行为特征（正是 GRPO 要修的那些）：
      - 贪心降低曼哈顿距离 → 在 U 型墙前反复撞墙 / 原地打转
      - 少量随机探索 → 偶尔能绕出去，偶尔超时
      - 极低概率输出非法方向 → 触发格式惩罚
    """

    def __init__(
        self,
        seed: int | None = None,
        invalid_rate: float = 0.02,
        explore_rate: float = 0.12,
        turn_rate: float = 0.8,
        latency: tuple[float, float] = (0.12, 0.35),
    ):
        self.rng = random.Random(seed)
        self.invalid_rate = invalid_rate
        self.explore_rate = explore_rate
        self.turn_rate = turn_rate
        self.latency = latency
        self._prev_pos: tuple[int, int] | None = None
        self._prev_action: str | None = None

    def act(self, obs: Observation) -> Step:
        t0 = time.perf_counter()
        direction = self._choose(obs)
        self._prev_pos, self._prev_action = obs.pos, direction

        text = format_tool_call(direction) if direction in ACTIONS else "I will try to move."
        action = direction if direction in ACTIONS else None
        latency = self.rng.uniform(*self.latency) - (time.perf_counter() - t0)
        return Step(
            text=text,
            action=action,
            tokens=[0] * max(8, len(text) // 4),
            logprob=round(-abs(self.rng.gauss(0.35, 0.5)), 3),
            latency=max(0.0, latency),
        )

    def _choose(self, obs: Observation) -> str:
        roll = self.rng.random()
        if roll < self.invalid_rate:
            return self.rng.choice(["northeast", "up-left", "jump", ""])
        if roll < self.invalid_rate + self.explore_rate:
            return self.rng.choice(list(ACTIONS))

        # 上一步撞墙（位置没变）→ 大概率换个方向，避免整段轨迹卡死不动，
        # 但依然只按曼哈顿贪心选，所以绕不出 U 型墙。
        if (
            self._prev_pos == obs.pos
            and self._prev_action in ACTIONS
            and self.rng.random() < self.turn_rate
        ):
            options = [d for d in ACTIONS if d != self._prev_action]
            return self.rng.choice(options)

        return min(
            ACTIONS,
            key=lambda name: manhattan(self._target(obs.pos, name), obs.maze.goal),
        )

    @staticmethod
    def _target(pos: tuple[int, int], name: str) -> tuple[int, int]:
        dx, dy = ACTIONS[name]
        return (pos[0] + dx, pos[1] + dy)


# ---------------------------------------------------------------- 采集


def run_episode(
    maze: Maze,
    policy: Policy,
    *,
    ui: "UIServer | None" = None,
    delay: float = 0.0,
    episode: str | None = None,
    obs_style: str = "minimal",
    memory: int = 3,
    show_visited: bool = False,
) -> Trajectory:
    env = MazeEnv(maze, memory=memory)
    traj = Trajectory(maze=maze)

    if ui is not None:
        from env.ui import build_frame

        ui.publish(build_frame(
            maze, env.pos, path=maze.shortest_path(), mode="reset",
            info={"steps": 0}, episode=episode, turn=0,
        ))

    turn = 0
    while not env.done:
        turn += 1
        visited = tuple(traj.path[:-1]) if show_visited else ()
        prompt = render_observation(maze, env.pos, env.last_actions,
                                    style=obs_style, visited=visited)
        obs = Observation(prompt, maze, env.pos, list(env.last_actions), turn)
        step = policy.act(obs)

        applied = step.action if step.action in ACTIONS else "invalid"
        result = env.step(applied)
        traj.total_reward += result.reward
        traj.turns.append(Turn(
            turn=turn,
            prompt=prompt,
            text=step.text,
            action=applied,
            reward=result.reward,
            done=result.done,
            info=result.info,
            logprob=step.logprob,
            pos=result.pos,
            prompt_ids=step.prompt_ids,
            tokens=step.tokens,
            token_logprobs=step.token_logprobs,
        ))

        if ui is not None:
            from env.ui import build_frame

            ui.publish(build_frame(
                maze, env.pos, path=maze.shortest_path(), mode="live",
                info={"action": applied, "reward": round(result.reward, 3), **result.info},
                llm={
                    "text": step.text,
                    "action": step.action,
                    "tokens": len(step.tokens) if step.tokens else None,
                    "logprob": step.logprob,
                    "latency": round(step.latency, 3),
                    "finish": step.finish,
                },
                prompt=prompt,
                episode=episode,
                turn=turn,
            ))
        if delay:
            time.sleep(delay)

    traj.success = env.terminated and env.pos == maze.goal
    traj.truncated = env.truncated
    return traj


def run_group(
    maze: Maze,
    policy: "Policy",
    g: int,
    *,
    ui: "UIServer | None" = None,
    episode_prefix: str = "g",
    obs_style: str = "minimal",
    memory: int = 0,
    show_visited: bool = False,
    delay: float = 0.0,
) -> list[Trajectory]:
    """同步批量 rollout：G 条轨迹每轮一起生成一次。

    与 run_episode 的区别：所有存活轨迹共享一次 engine.generate_batch，
    这是把 G 路并发的延迟摊平的关键。返回 G 条 Trajectory。
    """
    envs = [MazeEnv(maze, memory=memory) for _ in range(g)]
    trajs = [Trajectory(maze=maze) for _ in range(g)]
    tags = [f"{episode_prefix}/g{i}" for i in range(g)]
    from env.ui import build_frame

    turn = 0
    while True:
        active = [i for i, e in enumerate(envs) if not e.done]
        if not active:
            break
        turn += 1
        obs_list = []
        for i in active:
            env = envs[i]
            visited = tuple(trajs[i].path[:-1]) if show_visited else ()
            prompt = render_observation(maze, env.pos, env.last_actions,
                                        style=obs_style, visited=visited)
            obs_list.append(Observation(prompt, maze, env.pos,
                                        list(env.last_actions), turn))

        batch_fn = getattr(policy, "act_batch", None)
        steps = batch_fn(obs_list) if batch_fn else [policy.act(o) for o in obs_list]

        for i, obs, step in zip(active, obs_list, steps):
            env = envs[i]
            applied = step.action if step.action in ACTIONS else "invalid"
            result = env.step(applied)
            traj = trajs[i]
            traj.total_reward += result.reward
            traj.turns.append(Turn(
                turn=turn, prompt=obs.prompt, text=step.text,
                action=applied, reward=result.reward, done=result.done,
                info=result.info, logprob=step.logprob, pos=result.pos,
                prompt_ids=step.prompt_ids, tokens=step.tokens,
                token_logprobs=step.token_logprobs,
            ))
            if ui is not None:
                ui.publish(build_frame(
                    maze, env.pos, path=maze.shortest_path(), mode="live",
                    info={"action": applied, "reward": round(result.reward, 3),
                          **result.info},
                    llm={
                        "text": step.text,
                        "action": step.action,
                        "tokens": len(step.tokens) if step.tokens else None,
                        "logprob": step.logprob,
                        "latency": round(step.latency, 3),
                        "finish": step.finish,
                    },
                    prompt=obs.prompt,
                    episode=tags[i],
                    turn=turn,
                ))
        if delay:
            time.sleep(delay)

    for i, env in enumerate(envs):
        trajs[i].success = env.terminated and env.pos == maze.goal
        trajs[i].truncated = env.truncated
        if ui is not None:
            t = trajs[i]
            ui.publish(build_frame(
                maze, env.pos, path=t.path, mode="done",
                info={"success": t.success,
                      "total_reward": round(t.total_reward, 2),
                      "steps": t.steps, "reason": "done"},
                episode=tags[i], turn=turn,
            ))
    return trajs


def main() -> None:
    from env.maze import LEVELS, generate

    for level in LEVELS:
        stats = []
        for seed in range(5):
            maze = generate(level=level, seed=seed)
            traj = run_episode(maze, MockPolicy(seed=seed))
            stats.append(traj.summary())
        ok = sum(s["success"] for s in stats)
        rew = sum(s["total_reward"] for s in stats) / len(stats)
        steps = sum(s["steps"] for s in stats) / len(stats)
        print(f"{level}: success={ok}/5  avg_reward={rew:7.2f}  avg_steps={steps:5.1f}")


if __name__ == "__main__":
    main()
