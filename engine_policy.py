"""EnginePolicy — 用 fork 出来的推理引擎跑真实 rollout。

    engine = RolloutEngine(model, tokenizer, max_batch=8, max_seq_len=768)
    policy = EnginePolicy(engine, tokenizer)
    traj = run_episode(maze, policy, ui=ui)

`run_episode` / UI / 轨迹记录 与 MockPolicy 完全共用，换的只是 act()。
"""

from __future__ import annotations

import time

from engine_rl import RolloutEngine, Sampling
from prompts import render_prompt
from rollout import Observation, Step, parse_action


class EnginePolicy:
    def __init__(
        self,
        engine: RolloutEngine,
        tokenizer,
        sampling: Sampling | None = None,
        *,
        enable_thinking: bool = False,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.sampling = sampling or Sampling(max_tokens=48, temperature=1.0)
        self.enable_thinking = enable_thinking
        self.calls = 0
        self.total_latency = 0.0
        self.total_tokens = 0

    def act(self, obs: Observation) -> Step:
        _, prompt_ids = render_prompt(self.tokenizer, obs.prompt,
                                      self.enable_thinking)
        t0 = time.perf_counter()
        sample = self.engine.generate(prompt_ids, self.sampling)
        latency = time.perf_counter() - t0

        text = self.tokenizer.decode(sample.token_ids, skip_special_tokens=True)
        action = parse_action(text)

        self.calls += 1
        self.total_latency += latency
        self.total_tokens += len(sample.token_ids)

        return Step(
            text=text,
            action=action,
            tokens=sample.token_ids,
            logprob=sample.mean_logprob,
            latency=latency,
            finish=sample.finish_reason,
        )

    # ---- 统计 ----

    @property
    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "avg_latency_ms": (1000 * self.total_latency / self.calls)
            if self.calls else 0.0,
            "avg_tokens": (self.total_tokens / self.calls) if self.calls else 0.0,
        }

    def reset_stats(self) -> None:
        self.calls = 0
        self.total_latency = 0.0
        self.total_tokens = 0


__all__ = ["EnginePolicy"]
