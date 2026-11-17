"""多步 GRPO 训练器。

单步训练器按 prompt 分组是安全的：一个 prompt 最多对应 G 条采样。
多步训练里同一个状态会在多个 turn 重复出现，如果仍只按 prompt 分组，
一个 group 可能包含几十上百个样本，attention 显存会爆掉。

这里改用：

    group key = (prompt_ids, episode, turn)

并额外提供 max_group_size 硬上限。超出上限的 group 会切成 chunk；
所有 chunk 的梯度先累积，最后只 optimizer.step() 一次。
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .forward import token_logprobs
from .loss import LossStats, grpo_loss
from .train import GRPOTrainer, Sample


class MultiStepGRPOTrainer(GRPOTrainer):
    def update(self, samples: list[Sample], *,
               max_group_size: int = 8) -> LossStats:
        if max_group_size < 1:
            raise ValueError("max_group_size must be >= 1")

        samples = [s for s in samples
                   if s.completion_ids and len(s.old_logps) == len(s.completion_ids)]
        if not samples:
            raise ValueError("empty batch (no usable samples)")

        self.optimizer.zero_grad(set_to_none=True)
        self.model.model.grad_checkpoint = self.cfg.grad_checkpoint

        groups: dict[tuple, list[Sample]] = defaultdict(list)
        for s in samples:
            key = (
                tuple(s.prompt_ids),
                s.meta.get("episode"),
                s.meta.get("turn"),
            )
            groups[key].append(s)

        # 即使 key 仍然撞上，也把单组限制在 max_group_size 以内。
        chunks: list[list[Sample]] = []
        for group in groups.values():
            for i in range(0, len(group), max_group_size):
                chunks.append(group[i:i + max_group_size])

        total_tokens = sum(len(s.completion_ids) for s in samples)
        collected = []
        try:
            for chunk in chunks:
                ids, p, lens, old, adv, mask = self._pad_group(chunk)
                with torch.no_grad():
                    with self.adapter.ref_context() as ref_model:
                        ref = token_logprobs(ref_model, ids, p, lens)
                new = token_logprobs(self.model, ids, p, lens)
                outlier = (new.detach() - old).abs() > self.cfg.max_logp_delta
                mask = mask * (~outlier).float()

                loss, _ = grpo_loss(
                    new.reshape(-1), old.reshape(-1), adv.reshape(-1),
                    ref.reshape(-1), mask.reshape(-1),
                    clip_eps=self.cfg.clip_eps, beta=self.cfg.beta,
                )
                # 跨 chunk 累积；按整个 update 的 token 数归一化。
                (loss * (mask.sum() / total_tokens)).backward()
                collected.append((new.detach(), old, adv, ref, mask))
        finally:
            self.model.model.grad_checkpoint = False

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.adapter.trainable_parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.updates += 1

        packed = [torch.cat([c[i].reshape(-1) for c in collected])
                  for i in range(5)]
        _, stats = grpo_loss(*packed, clip_eps=self.cfg.clip_eps,
                             beta=self.cfg.beta)
        stats.grad_norm = float(grad_norm)
        kept = sum(float(c[4].sum()) for c in collected)
        stats.drop_frac = 1.0 - kept / max(total_tokens, 1.0)
        return stats


def samples_from_multistep_trajectory(traj, advantage: float,
                                      episode: str | None = None) -> list[Sample]:
    """把一条多步轨迹摊平成 Sample；episode 用于多步分组。"""
    from .train import samples_from_trajectory

    return samples_from_trajectory(traj, advantage, episode=episode)
