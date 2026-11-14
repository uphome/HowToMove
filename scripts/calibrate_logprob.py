"""T2：引擎 logprob ↔ 训练 forward 重算的一致性校准。

这是整个项目最容易静默出错、又最致命的环节：
如果两侧 logprob 不一致，GRPO 的 ratio = exp(new - old) 在第 0 步就不等于 1，
梯度全是噪声，而且不会有任何报错。

对比的两条路径正是实际使用的：
    A. 引擎：Triton varlen flash prefill + eager decode + KV cache
    B. 训练：QWEN3_FLASH_ATTN=pytorch → SDPA，整条序列一次前向

门槛：max|Δ| < 1e-2，mean|Δ| < 1e-3，ratio ∈ [0.98, 1.02]

    python scripts/calibrate_logprob.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402,F401  (注入 engine/ 到 sys.path)

import torch  # noqa: E402

from engine_rl import RolloutEngine, Sampling, load_model  # noqa: E402
from env.maze import generate  # noqa: E402
from env.render import render_state  # noqa: E402
from prompts import render_prompt  # noqa: E402

def build_prompt(tokenizer, level: str = "L3", seed: int = 114) -> tuple[str, list[int]]:
    """构造和真实 rollout 完全一致的 prompt（同一个 prompts.render_prompt）。"""
    maze = generate(level=level, seed=seed)
    return render_prompt(tokenizer, render_state(maze, maze.start, []))


@torch.no_grad()
def recompute_logprobs(model, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """整条序列一次前向，返回生成部分每个 token 的 logprob。"""
    old_flash = os.environ.get("QWEN3_FLASH_ATTN")
    os.environ["QWEN3_FLASH_ATTN"] = "pytorch"
    try:
        logits = model(input_ids)
    finally:
        if old_flash is None:
            os.environ.pop("QWEN3_FLASH_ATTN", None)
        else:
            os.environ["QWEN3_FLASH_ATTN"] = old_flash

    gen = input_ids[0, prompt_len:]
    pred = logits[0, prompt_len - 1 : -1].float()
    log_probs = torch.log_softmax(pred, dim=-1)
    return log_probs.gather(-1, gen.unsqueeze(-1)).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(paths.MODEL_PATH))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--level", default="L3")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用，T2 需要 GPU（bf16 kernel 路径）")
        return 2

    print(f"[1/4] 加载模型 {args.model}")
    model, tokenizer, device = load_model(args.model, args.device)
    print(f"      device={device} dtype={next(model.parameters()).dtype}")

    prompt_text, prompt_ids = build_prompt(tokenizer, args.level, args.seed)
    print(f"[2/4] prompt: {len(prompt_ids)} tokens")

    engine = RolloutEngine(model, tokenizer, max_batch=1, max_seq_len=1024,
                           kv_ratio=0.5, device=device)
    try:
        print(f"[3/4] 引擎生成 {args.tokens} tokens（greedy）")
        sample = engine.generate(prompt_ids, Sampling(
            max_tokens=args.tokens, temperature=0.0))
        gen_ids = sample.token_ids
        old = torch.tensor(sample.logprobs, dtype=torch.float32)
        print(f"      gen={len(gen_ids)} tokens, finish={sample.finish_reason}")
        print(f"      ids={gen_ids[:12]}{'...' if len(gen_ids) > 12 else ''}")
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        print(f"      text={text[:120]!r}")
        if not gen_ids:
            print("FAIL: 引擎没有产出 token")
            return 1

        print("[4/4] 用 SDPA 训练路径重算 logprob")
        full = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=device)
        model.to(device)
        new = recompute_logprobs(model, full, len(prompt_ids)).cpu()
    finally:
        engine.close()

    n = min(len(old), len(new))
    old, new = old[:n], new[:n]
    diff = (new - old).abs()
    ratio = torch.exp(new - old)

    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rmin, rmax = float(ratio.min()), float(ratio.max())
    ok = max_abs < 1e-2 and mean_abs < 1e-3 and 0.98 <= rmin and rmax <= 1.02

    print()
    print(f"  tokens      : {n}")
    print(f"  max |Δ|     : {max_abs:.6f}   (门槛 1e-2)")
    print(f"  mean |Δ|    : {mean_abs:.6f}   (门槛 1e-3)")
    print(f"  ratio range : [{rmin:.6f}, {rmax:.6f}]   (门槛 [0.98, 1.02])")
    print(f"  old[0:4]    : {old[:4].tolist()}")
    print(f"  new[0:4]    : {new[:4].tolist()}")
    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
