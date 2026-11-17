"""Chat-template policy for explicit message/tool-result agents."""

from __future__ import annotations

import paths  # noqa: F401

from engine_rl import RolloutEngine, Sampling
from prompts import TOOLS
from rollout import Step, parse_action


class ChatPolicy:
    """Generate one assistant tool call from a full messages list."""

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
        self.sampling = sampling or Sampling(max_tokens=48, temperature=0.0)
        self.enable_thinking = enable_thinking
        self.calls = 0
        self.last_prompt_text = ""

    def act(self, messages: list[dict]) -> Step:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=TOOLS,
            enable_thinking=self.enable_thinking,
        )
        self.last_prompt_text = text
        ids = self.tokenizer.encode(text)
        samples = self.engine.generate_batch([ids], self.sampling)
        sample = samples[0]
        output = self.tokenizer.decode(sample.token_ids, skip_special_tokens=True)
        self.calls += 1
        return Step(
            text=output,
            action=parse_action(output),
            tokens=sample.token_ids,
            logprob=sample.mean_logprob,
            finish=sample.finish_reason,
            prompt_ids=ids,
            token_logprobs=sample.logprobs,
        )


__all__ = ["ChatPolicy"]
