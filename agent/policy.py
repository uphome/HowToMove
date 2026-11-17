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
        self.last_prompt_texts: list[str] = []

    def act(self, messages: list[dict]) -> Step:
        return self.act_batch([messages])[0]

    def act_batch(self, messages_batch: list[list[dict]]) -> list[Step]:
        if not messages_batch:
            return []
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=TOOLS,
                enable_thinking=self.enable_thinking,
            )
            for messages in messages_batch
        ]
        ids_batch = [self.tokenizer.encode(p) for p in prompts]
        self.last_prompt_texts = prompts
        self.last_prompt_text = prompts[-1]
        samples = self.engine.generate_batch(ids_batch, self.sampling)
        if len(samples) != len(ids_batch):
            raise RuntimeError(
                f"engine returned {len(samples)} samples for "
                f"{len(ids_batch)} prompts")
        steps = []
        for ids, sample in zip(ids_batch, samples):
            output = self.tokenizer.decode(
                sample.token_ids, skip_special_tokens=True)
            steps.append(Step(
                text=output,
                action=parse_action(output),
                tokens=sample.token_ids,
                logprob=sample.mean_logprob,
                finish=sample.finish_reason,
                prompt_ids=ids,
                token_logprobs=sample.logprobs,
            ))
        self.calls += len(steps)
        return steps


__all__ = ["ChatPolicy"]
