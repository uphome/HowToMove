"""RolloutEngine — RL 训练用的进程内批量推理接口。

只依赖 ``engine/qwen3`` 核心（模型 / PagedKV / Scheduler / sampling），
不经过 HTTP，也不带 server / bench 等外围脚本：

    engine = RolloutEngine(model, tokenizer, max_batch=8, max_seq_len=768)
    samples = engine.generate_batch(prompt_ids_list, Sampling(max_tokens=48))
    # samples[i].token_ids / .logprobs / .finish_reason

设计要点：
  - 同一个 Scheduler + PagedKVCachePool，一次 generate_batch 内连续批处理；
  - 模型对象由训练侧注入，optimizer.step() 后下一次 rollout 自动是新策略；
  - logprob 取**温度/截断之前**的原始 log_softmax，训练侧直接当 old_logps；
  - top_k=0 / top_p=1 时走 torch.multinomial，避免对 15 万词表做 full topk。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from qwen3 import KVCachePool, Qwen3Config, Qwen3ForCausalLM, Scheduler
from qwen3 import load_weights_from_safetensors
from qwen3.request import Request
from qwen3.sampling import sample_tokens

BLOCK_SIZE = 16
EOS_TOKEN_IDS = (151645, 151643)   # <|im_end|> / <|endoftext|>


@dataclass
class Sampling:
    max_tokens: int = 48
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: int | tuple[int, ...] = EOS_TOKEN_IDS
    timeout: float = 600.0          # 保留字段；同步接口不做超时中断

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


def _is_eos(token: int, eos) -> bool:
    eos_ids = eos if isinstance(eos, (tuple, list, set)) else (eos,)
    return token in eos_ids


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
        self.runner = runner          # 预留：CUDA Graph runner 暂未接入
        self._next_id = 0

        blocks = _blocks_for(max_seq_len) * max_batch
        blocks = _cap_blocks(blocks, model.config, BLOCK_SIZE, self.device,
                             kv_ratio)
        self.pool = KVCachePool(
            num_blocks=blocks,
            num_layers=model.config.num_hidden_layers,
            block_size=BLOCK_SIZE,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.config.head_dim,
            device=self.device,
            dtype=next(model.parameters()).dtype,
            max_seq_len=max_seq_len,
            max_requests=max(8, max_batch),
        )
        self.scheduler = Scheduler(self.pool, batch_size=max_batch)

    # ---- 生命周期 ----

    def close(self) -> None:
        """释放 KV 池 / 调度器引用（模型仍由调用方持有）。"""
        self.scheduler = None
        self.pool = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self) -> "RolloutEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 推理 ----

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids: list[list[int]],
        sampling: Sampling | None = None,
    ) -> list[Sample]:
        """并发提交 N 个 prompt，等全部完成后按输入顺序返回。

        同步接口：调用方（训练循环）本来就串行等待整组 rollout，
        不再需要后台线程 / Job / Event。批内由 Scheduler 连续批处理。
        """
        if not prompt_ids:
            return []
        sampling = sampling or Sampling()
        if len(prompt_ids) > self.max_batch:
            raise ValueError(
                f"batch {len(prompt_ids)} > max_batch {self.max_batch}")
        if sampling.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        total_blocks = 0
        for ids in prompt_ids:
            if not ids:
                raise ValueError("empty prompt")
            if len(ids) + sampling.max_tokens > self.max_seq_len:
                raise ValueError(
                    f"prompt {len(ids)} + max_tokens {sampling.max_tokens} "
                    f"> max_seq_len {self.max_seq_len}")
            total_blocks += _blocks_for(len(ids) + sampling.max_tokens)
        if total_blocks > self.pool.free_count:
            raise RuntimeError(
                f"KV 池容量不足: 本批需要 {total_blocks} 块，"
                f"剩余 {self.pool.free_count} 块（可调大 max_seq_len / kv_ratio）")

        reqs: list[Request] = []
        finished: dict[int, Request] = {}
        try:
            for ids in prompt_ids:
                req = Request(
                    request_id=self._next_id,
                    input_ids=torch.tensor([ids], dtype=torch.long,
                                           device=self.device),
                    max_new_tokens=sampling.max_tokens,
                    eos_token_id=sampling.eos_token_id,
                )
                self._next_id += 1
                self.scheduler.add(req)
                reqs.append(req)

            while self.scheduler.has_pending():
                batch = self.scheduler.schedule()
                if batch is None:
                    break
                logits = self._forward_logits(batch)
                next_tokens, next_logprobs = self._sample(logits, sampling)
                self.scheduler.on_step_done(batch, next_tokens, next_logprobs)

                for req in batch.requests:
                    if req.state == "finished":
                        finished[req.request_id] = req
                        # 防止 finished 列表长期持有 input_ids/KV 句柄
                        if req in self.scheduler.finished:
                            self.scheduler.finished.remove(req)
        except Exception:
            # 异常时归还本批所有请求的 KV 块与行槽位，保持池可复用
            for req in reqs:
                if req.kv_cache is not None:
                    req.kv_cache.free()
                    req.kv_cache = None
            self.scheduler = Scheduler(self.pool, batch_size=self.max_batch)
            raise

        results: list[Sample] = []
        for req in reqs:
            done = finished.get(req.request_id)
            if done is None:
                results.append(Sample([], [], "error", req.request_id,
                                      "request did not finish"))
                continue
            reason = ("stop" if done.output_ids
                      and _is_eos(done.output_ids[-1], sampling.eos_token_id)
                      else "length")
            results.append(Sample(
                token_ids=list(done.output_ids),
                logprobs=list(done.output_logprobs),
                finish_reason=reason,
                request_id=done.request_id,
            ))
        return results

    def generate(self, prompt_ids: list[int],
                 sampling: Sampling | None = None) -> Sample:
        return self.generate_batch([prompt_ids], sampling)[0]

    # ---- 内部 ----

    @torch.no_grad()
    def _forward_logits(self, batch) -> torch.Tensor:
        """只算每个请求最后一个位置的 logits，返回 (B, vocab)。

        prefill 不物化 (B,S,V)：S=256 / V=151936 / bf16 时 622MB → 2.4MB。
        """
        input_ids = batch.build_input_ids()
        caches = batch.build_kv_caches()
        last = torch.tensor(batch.input_lens, dtype=torch.long,
                            device=input_ids.device) - 1
        logits = self.model(
            input_ids,
            kv_cache=caches,
            input_lens=batch.input_lens,
            last_positions=last,
        )
        return logits[:, -1, :]

    @staticmethod
    def _sample(logits: torch.Tensor, sampling: Sampling):
        """返回 (B,1) token 与 (B,) 原始分布 logprob。"""
        raw_logprobs = F.log_softmax(logits.float(), dim=-1)

        if sampling.temperature <= 0:
            tokens = logits.argmax(dim=-1, keepdim=True)
        elif sampling.top_k <= 0 and sampling.top_p >= 1.0:
            # 全词表精确采样；避免 sample_tokens 的 topk=0 回退做 full sort
            probs = F.softmax(logits.float() / float(sampling.temperature),
                              dim=-1)
            tokens = torch.multinomial(probs, num_samples=1)
        else:
            tokens = sample_tokens(
                logits,
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                top_p=sampling.top_p,
            ).long()

        token_logprobs = raw_logprobs.gather(-1, tokens).squeeze(-1)
        return tokens, token_logprobs

    # ---- 统计 ----

    @property
    def pool_stats(self) -> dict:
        if self.pool is None:
            return {"blocks": 0, "free": 0,
                    "tokens_per_block": BLOCK_SIZE, "capacity_tokens": 0}
        return {
            "blocks": self.pool.total_blocks,
            "free": self.pool.free_count,
            "tokens_per_block": BLOCK_SIZE,
            "capacity_tokens": self.pool.total_blocks * BLOCK_SIZE,
        }


def _cap_blocks(blocks: int, config, block_size: int, device, ratio: float) -> int:
    """按剩余显存裁剪池大小，避免小卡启动即 OOM。"""
    if device.type != "cuda":
        return blocks
    torch.cuda.empty_cache()
    free_bytes = torch.cuda.mem_get_info(device)[0]
    per_block = (config.num_hidden_layers * config.num_key_value_heads
                 * block_size * config.head_dim * 2 * 2)
    capacity = int(free_bytes * ratio / per_block)
    return max(1, min(blocks, capacity))


def load_model(model_path: str, device: str | torch.device | None = None):
    """加载模型 + tokenizer（与引擎核心同一套权重加载路径）。"""
    from transformers import AutoTokenizer

    if device is None:
        device = ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
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
    # CUDA Graph 只影响延迟，不影响正确性；RL 先走 eager decode。
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
