"""GRPO loss（照抄 TRL 的公式，token-level 归一化）。

    ratio = exp(new_logps - old_logps)          old 必须 detach
    L_pg  = -min(ratio * A, clip(ratio) * A)
    KL    = ref_logps - new_logps               （k1，无偏但可为负）
    loss  = (L_pg + beta * KL).sum() / mask.sum()

token-level 归一化（DAPO 做法）而不是"先每条序列平均再平均"：
后者会让短轨迹的 token 权重被人为放大。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossStats:
    loss: float
    pg: float
    kl: float
    ratio_mean: float
    ratio_min: float
    ratio_max: float
    clip_frac: float
    tokens: int
    drop_frac: float = 0.0
    grad_norm: float = 0.0


def grpo_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    ref_logps: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    clip_eps: float = 0.2,
    beta: float = 0.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, LossStats]:
    """所有张量都是 1D（已按样本拼接）。advantages 已按 token 广播。"""
    if mask is None:
        mask = torch.ones_like(new_logps, dtype=torch.float32)
    else:
        mask = mask.float()

    new = new_logps.float()
    old = old_logps.detach().float()
    adv = advantages.float()

    # 防止 exp 溢出：bf16 引擎日志与 fp32 重算的差通常在 1e-3 量级，
    # 但推理出错时可能很大，夹一下避免 NaN
    log_ratio = torch.clamp(new - old, -20.0, 20.0)
    ratio = torch.exp(log_ratio)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg = -torch.min(surr1, surr2)

    if ref_logps is not None:
        kl = ref_logps.float() - new
    else:
        kl = torch.zeros_like(new)

    n = mask.sum().clamp(min=1.0)
    loss = ((pg + beta * kl) * mask).sum() / n

    with torch.no_grad():
        clipped = (ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)
        stats = LossStats(
            loss=float(loss.detach()),
            pg=float(((pg * mask).sum() / n).detach()),
            kl=float(((kl * mask).sum() / n).detach()),
            ratio_mean=float((ratio * mask).sum() / n),
            ratio_min=float(ratio.min()),
            ratio_max=float(ratio.max()),
            clip_frac=float((clipped.float() * mask).sum() / n),
            tokens=int(mask.sum()),
        )
    return loss, stats


__all__ = ["grpo_loss", "LossStats"]
