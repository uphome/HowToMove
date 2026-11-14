"""训练侧前向：只对需要的 token 位置算 logprob，支持组内批处理。

显存要点：
  1. 不物化 (B, S, vocab)。先拿 hidden states，只 gather 需要的位置，
     再分块过 lm_head 并立即算出目标 token 的 logprob。
  2. 注意力必须走 PyTorch 标准实现：引擎的 Triton varlen flash kernel
     是纯推理的，没有 backward。
  3. 组内批处理 + 梯度检查点：B=8 时激活会到 ~2.8GB，4GB 卡必须重算。
"""

from __future__ import annotations

import contextlib
import os

import torch

PAD_ID = 0


@contextlib.contextmanager
def _force_pytorch_attention():
    old = os.environ.get("QWEN3_FLASH_ATTN")
    os.environ["QWEN3_FLASH_ATTN"] = "pytorch"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("QWEN3_FLASH_ATTN", None)
        else:
            os.environ["QWEN3_FLASH_ATTN"] = old


def hidden_states(model, input_ids: torch.Tensor,
                  input_lens: list[int] | None = None) -> torch.Tensor:
    """(B, S) -> (B, S, H)。不经过 lm_head，避免 (B,S,V)。"""
    with _force_pytorch_attention():
        return model.model(input_ids, input_lens=input_lens)


def gather_logprobs(
    model,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    targets: torch.Tensor,
    chunk: int = 64,
) -> torch.Tensor:
    """在 hidden 的指定位置算 targets 的 logprob，形状 (B, P)。

    分块 + logsumexp，避免一次物化 (B*P, V)：
        单块 (chunk, V) fp32，chunk=64 时约 39MB。
    """
    b, s, h = hidden.shape
    p = positions.shape[1]
    rows = torch.arange(b, device=hidden.device).unsqueeze(1)
    picked = hidden[rows, positions].reshape(b * p, h)     # (B*P, H)
    flat_t = targets.reshape(b * p)

    out = []
    for i in range(0, picked.shape[0], chunk):
        logits = model.lm_head(picked[i:i + chunk]).float()
        lp = (logits.gather(-1, flat_t[i:i + chunk, None]).squeeze(-1)
              - torch.logsumexp(logits, dim=-1))
        out.append(lp)
    return torch.cat(out).view(b, p)


def token_logprobs(
    model,
    input_ids: torch.Tensor,
    prompt_len: int,
    lengths: list[int] | None = None,
    chunk: int = 64,
) -> torch.Tensor:
    """返回 prompt 之后每个 token 的 logprob，形状 (B, T)。

    input_ids 右 padding；lengths 给出每个样本的 completion 实际长度。
    padding 位置的返回值是噪声，调用方必须用 mask 排除。
    """
    b, s = input_ids.shape
    if s <= prompt_len:
        return input_ids.new_zeros((b, 0), dtype=torch.float32)
    t = s - prompt_len
    lens = lengths or [t] * b
    positions = torch.arange(prompt_len - 1, s - 1,
                             device=input_ids.device).unsqueeze(0).expand(b, -1)
    targets = input_ids[:, prompt_len:]
    hidden = hidden_states(model, input_ids,
                           input_lens=[prompt_len + n for n in lens])
    return gather_logprobs(model, hidden, positions, targets, chunk=chunk)


def completion_mask(lengths: list[int], device=None) -> torch.Tensor:
    """(B, T) 的 0/1 mask。"""
    t = max(lengths)
    idx = torch.arange(t, device=device).unsqueeze(0)
    lens = torch.tensor(lengths, device=device).unsqueeze(1)
    return (idx < lens).float()
