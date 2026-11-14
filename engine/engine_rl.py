"""RolloutEngine — RL 训练用的进程内批量推理接口。

和 server.py 的区别只有两点：
    1. 不走 HTTP：直接提交 Job 到同一个 Scheduler，拿回 token ids + logprobs；
    2. 显存按 RL 场景配置（短 prompt、G 路并发、不预留 2048 输入余量）。

模型对象由调用方注入：训练侧和推理侧共用**同一个实例**，
optimizer.step() 之后下一次 rollout 天然就是新策略（无需权重同步）。

    engine = RolloutEngine(model, tokenizer, max_batch=8, max_seq_len=768)
    samples = engine.generate_batch(prompt_ids_list, Sampling(max_tokens=48))
    # samples[i].token_ids / .logprobs / .finish_reason
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace

import torch

from qwen3 import KVCachePool

from server import BLOCK_SIZE, EOS_TOKEN_ID, Engine, Job


@dataclass
class Sampling:
    max_tokens: int = 48
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: int | tuple[int, ...] = (EOS_TOKEN_ID, 151643)
    timeout: float = 600.0

    def greedy(self) -> "Sampling":
        return replace(self, temperature=0.0)


@dataclass
class Sample:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str = "stop"
    request_id: int = -1
    error: str | None = None

    @property
    def mean_logprob(self) -> float:
        return sum(self.logprobs) / len(self.logprobs) if self.logprobs else 0.0


def _blocks_for(tokens: int) -> int:
    return math.ceil(tokens / BLOCK_SIZE)


class RolloutEngine:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_batch: int = 8,
        max_seq_len: int = 768,
        kv_ratio: float = 0.55,
        runner=None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.device = device or next(model.parameters()).device

        blocks = _blocks_for(max_seq_len) * max_batch
        blocks = _cap_blocks(blocks, model.config, BLOCK_SIZE, self.device, kv_ratio)
        self.pool = KVCachePool(
            num_blocks=blocks,
            num_layers=model.config.num_hidden_layers,
            block_size=BLOCK_SIZE,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            device=self.device,
            dtype=next(model.parameters()).dtype,
            max_seq_len=max_seq_len,
        )
        self.runner = runner
        self._engine = Engine(model, self.pool, runner, batch_size=max_batch)
        self._engine.start()
        self._id_lock = threading.Lock()

    # ---- 生命周期 ----

    def close(self) -> None:
        self._engine.stop()

    def __enter__(self) -> "RolloutEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 推理 ----

    def generate_batch(
        self,
        prompt_ids: list[list[int]],
        sampling: Sampling | None = None,
    ) -> list[Sample]:
        """并发提交 N 个 prompt，等全部完成后按输入顺序返回。"""
        if not prompt_ids:
            return []
        sampling = sampling or Sampling()
        if len(prompt_ids) > self.max_batch:
            raise ValueError(
                f"batch {len(prompt_ids)} > max_batch {self.max_batch}")

        jobs: list[Job] = []
        for ids in prompt_ids:
            if not ids:
                raise ValueError("empty prompt")
            if len(ids) + sampling.max_tokens > self.max_seq_len:
                raise ValueError(
                    f"prompt {len(ids)} + max_tokens {sampling.max_tokens} "
                    f"> max_seq_len {self.max_seq_len}")
            with self._id_lock:
                rid = self._engine._next_id
                self._engine._next_id += 1
            job = Job(
                request_id=rid,
                input_ids=torch.tensor([ids], dtype=torch.long, device=self.device),
                max_new_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
                eos_token_id=sampling.eos_token_id,
            )
            self._engine.submit(job)
            jobs.append(job)

        results: list[Sample] = []
        for job in jobs:
            if not job.done.wait(sampling.timeout):
                results.append(Sample([], [], "timeout", job.request_id,
                                      "rollout timeout"))
                continue
            if job.error:
                results.append(Sample([], [], "error", job.request_id, job.error))
                continue
            results.append(Sample(
                token_ids=list(job.output_ids),
                logprobs=list(job.logprobs),
                finish_reason=job.finish_reason,
                request_id=job.request_id,
            ))
        return results

    def generate(
        self, prompt_ids: list[int], sampling: Sampling | None = None
    ) -> Sample:
        return self.generate_batch([prompt_ids], sampling)[0]

    # ---- 统计 ----

    @property
    def pool_stats(self) -> dict:
        return {
            "blocks": self.pool.total_blocks,
            "free": self.pool.free_count,
            "tokens_per_block": BLOCK_SIZE,
            "capacity_tokens": self.pool.total_blocks * BLOCK_SIZE,
        }


def _cap_blocks(blocks: int, config, block_size: int, device, ratio: float) -> int:
    """按剩余显存裁剪池大小，避免 4GB 卡启动即 OOM。"""
    if device.type != "cuda":
        return blocks
    from server import kv_bytes_per_block

    torch.cuda.empty_cache()
    free_bytes = torch.cuda.mem_get_info(device)[0]
    per_block = kv_bytes_per_block(config, block_size)
    capacity = int(free_bytes * ratio / per_block)
    return max(1, min(blocks, capacity))


def load_model(model_path: str, device: str | torch.device | None = None):
    """加载模型 + tokenizer（与 server.py 同一套权重加载路径）。"""
    from transformers import AutoTokenizer

    from qwen3 import Qwen3Config, Qwen3ForCausalLM, load_weights_from_safetensors

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = Qwen3Config.from_pretrained(model_path)
    model = Qwen3ForCausalLM(config)
    load_weights_from_safetensors(model, model_path, device, dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer, device


def build(
    model_path: str,
    *,
    max_batch: int = 8,
    max_seq_len: int = 768,
    kv_ratio: float = 0.55,
    use_graph: bool = False,
    device: str | torch.device | None = None,
) -> RolloutEngine:
    """独立使用（不训练）时的完整装配。"""
    model, tokenizer, device = load_model(model_path, device)
    # M1 先走 eager decode：CUDA Graph 只影响延迟，不影响正确性，
    # 等 T2/T3 的 ratio 校准稳定后再开（M3 优化项）。
    runner = None if not use_graph else None
    return RolloutEngine(
        model, tokenizer,
        max_batch=max_batch, max_seq_len=max_seq_len,
        kv_ratio=kv_ratio, runner=runner, device=device,
    )


__all__ = [
    "RolloutEngine",
    "Sample",
    "Sampling",
    "build",
    "load_model",
]
