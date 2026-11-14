"""最小可用 GRPO 训练器（组内批处理 + 梯度检查点）。

一次 update 的输入是一批 Sample（prompt + completion + old_logps + advantage）。
按 prompt 分组（同一状态的所有采样共享 prompt），组内右 padding 成 (B, S)
一次前向 —— 单样本循环在 4GB 卡上要 7s/组，批处理 + 重算后约 0.6s。

显存：B=8, S≈340 时，不重算要 ~2.8GB 激活，重算后 ~300MB。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .adapter import TrainAdapter
from .forward import PAD_ID, completion_mask, token_logprobs
from .loss import LossStats, grpo_loss


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    old_logps: list[float]
    advantage: float
    meta: dict = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def full_ids(self) -> list[int]:
        return list(self.prompt_ids) + list(self.completion_ids)


@dataclass
class TrainConfig:
    lr: float = 1e-4
    clip_eps: float = 0.2
    beta: float = 0.0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    grad_checkpoint: bool = True
    # 引擎 bf16 logits 与训练重算的差通常 <1e-2；超过这个阈值视为数值离群
    # （不是策略变化），从 loss 里剔除，避免 exp() 放大成噪声梯度。
    max_logp_delta: float = 0.5


class GRPOTrainer:
    def __init__(self, adapter: TrainAdapter, config: TrainConfig | None = None,
                 device: torch.device | None = None):
        self.adapter = adapter
        self.model = adapter.model
        self.cfg = config or TrainConfig()
        self.device = device or next(self.model.parameters()).device
        self.optimizer = torch.optim.AdamW(adapter.param_groups(self.cfg.lr))
        self.updates = 0

    # ---- 组批 ----

    def _pad_group(self, group: list[Sample]):
        prompt = list(group[0].prompt_ids)
        p = len(prompt)
        lens = [len(s.completion_ids) for s in group]
        t = max(lens)
        b = len(group)
        ids = torch.full((b, p + t), PAD_ID, dtype=torch.long, device=self.device)
        ids[:, :p] = torch.tensor(prompt, dtype=torch.long, device=self.device)
        old = torch.zeros(b, t, dtype=torch.float32, device=self.device)
        adv = torch.zeros(b, t, dtype=torch.float32, device=self.device)
        for i, s in enumerate(group):
            n = len(s.completion_ids)
            ids[i, p:p + n] = torch.tensor(s.completion_ids, dtype=torch.long,
                                           device=self.device)
            old[i, :n] = torch.tensor(s.old_logps, dtype=torch.float32,
                                      device=self.device)
            adv[i, :n] = s.advantage
        mask = completion_mask(lens, device=self.device)
        return ids, p, lens, old, adv, mask

    # ---- 一次更新 ----

    def update(self, samples: list[Sample]) -> LossStats:
        samples = [s for s in samples
                   if s.completion_ids and len(s.old_logps) == len(s.completion_ids)]
        if not samples:
            raise ValueError("empty batch (no usable samples)")

        self.optimizer.zero_grad(set_to_none=True)
        self.model.model.grad_checkpoint = self.cfg.grad_checkpoint
        groups: dict[tuple, list[Sample]] = defaultdict(list)
        for s in samples:
            groups[tuple(s.prompt_ids)].append(s)

        total_tokens = sum(len(s.completion_ids) for s in samples)
        collected = []
        try:
            for group in groups.values():
                ids, p, lens, old, adv, mask = self._pad_group(group)
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
                # grpo_loss 内部按 mask.sum() 归一化；乘回 token 占比后，
                # 多组累积等价于"整批 token 一次归一化"
                (loss * (mask.sum() / total_tokens)).backward()
                collected.append((new.detach(), old, adv, ref, mask))
        finally:
            self.model.model.grad_checkpoint = False

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.adapter.trainable_parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.updates += 1

        packed = [torch.cat([c[i].reshape(-1) for c in collected]) for i in range(5)]
        _, stats = grpo_loss(*packed, clip_eps=self.cfg.clip_eps,
                             beta=self.cfg.beta)
        stats.grad_norm = float(grad_norm)
        total = sum(c[4].numel() for c in collected)
        kept = sum(float(c[4].sum()) for c in collected)
        stats.drop_frac = 1.0 - kept / max(total, 1.0)
        return stats

    # ---- 存档 ----

    def save(self, path: str) -> None:
        torch.save({
            "adapter": self.adapter.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "config": self.cfg.__dict__,
        }, path)

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(blob["adapter"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self.updates = blob.get("updates", 0)


def samples_from_trajectory(traj, advantage: float) -> list[Sample]:
    """把一条多步轨迹摊平成训练样本：每一步一个 (prompt, completion, old_logps)。

    advantage 是轨迹级标量，广播到该轨迹的所有 token（GRPO 的标准做法）。
    """
    out = []
    for t in getattr(traj, "turns", []):
        if not (t.token_logprobs and t.prompt_ids and t.tokens):
            continue
        if len(t.token_logprobs) != len(t.tokens):
            continue
        out.append(Sample(
            prompt_ids=list(t.prompt_ids),
            completion_ids=list(t.tokens),
            old_logps=[float(x) for x in t.token_logprobs],
            advantage=float(advantage),
            meta={"turn": t.turn, "episode": None},
        ))
    return out


def group_advantages(rewards: list[float], eps: float = 1e-4
                     ) -> tuple[list[float], bool]:
    """组内标准化。返回 (advantages, degenerate)。

    degenerate=True 表示这组没有区分度（全成功/全失败/全一样），
    advantage 全 0，调用方应跳过本次更新 —— 否则只是在噪声上更新。
    """
    n = len(rewards)
    if n < 2:
        return [0.0] * n, True
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < 1e-6:
        return [0.0] * n, True
    return [(r - mean) / (std + eps) for r in rewards], False


__all__ = ["GRPOTrainer", "Sample", "TrainConfig", "group_advantages",
           "samples_from_trajectory"]
