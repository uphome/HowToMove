# FORK.md — 引擎 fork 说明

本目录是 [`Qwen3-engine-from-scratch`](https://github.com/uphomer/Qwen3-engine-from-scratch)
的 vendored fork，作为 AgenticRL 项目的一部分维护。

| | |
|---|---|
| 上游 commit | `ad04f0bff78e37b78c597a16b92ae6d24ab4b548` |
| 上游信息 | `feat: 新增 OpenAI 兼容 HTTP API 服务（server.py）` |
| fork 时间 | 2026-11-14 |
| 剔除内容 | `.git/` `logs/` `__pycache__/` `trace.json` `profile_summary.txt` |

导入方式：`AgenticRL/paths.py` 把本目录插入 `sys.path`，因此引擎内部的
`from qwen3 import ...` / `import server` **一行都不用改**。

---

## 改动清单

### 新增

| 文件 | 作用 |
|---|---|
| `engine_rl.py` | 进程内批量推理接口 `RolloutEngine.generate_batch()`，返回 token ids + logprobs，不经 HTTP |
| `FORK.md` | 本文件 |

### 修改

| 文件 | 位置 | 改动 |
|---|---|---|
| `server.py` | `sample_one()` | 返回 `(token, logprob)`；logprob 取**温度缩放/top-k/top-p 之前**的原始分布 |
| `server.py` | `Job` | 新增 `logprobs: list[float]` |
| `server.py` | `Engine._step()` | 收集每步 logprob；`no_grad`；prefill 只算最后位置 logits |
| `server.py` | `_run_generation()` / `_handle_chat()` / `_handle_completion()` | 响应新增非标准字段 `token_ids` / `logprobs` |
| `qwen3/request.py` | `append()` | 新增 `logprob` 参数；新增 `output_logprobs` 字段 |
| `qwen3/request.py` | `is_finished()` | 支持多 EOS（151645 `<|im_end|>` 与 151643 `<|endoftext|>`） |
| `qwen3/scheduler.py` | `on_step_done()` | 透传 `next_logprobs` |
| `qwen3/model.py` | `Qwen3ForCausalLM.forward()` | 新增 `last_positions`：只对指定位置做 `lm_head` |

---

## 为什么这么改

### 1. logprob 取原始分布，不取采样分布

训练侧重算的是 `log_softmax(raw_logits)`。如果引擎返回的是温度/截断之后的
分布，`ratio = exp(new - old)` 在第 0 步就不等于 1，梯度全是噪声。
所以：**采样可以用温度，记录必须用原始分布**。温度只负责探索。

### 2. `no_grad` 是修 bug，不是优化

上游 `Engine._step()` 没有包 `torch.no_grad()`。推理不走 backward，
但 PyTorch 仍然为每次前向建反图，28 层激活全部常驻。

实测（8 路 × 256 token，4GB 卡）：

| 轮次 | 修复前 `memory_allocated` | 修复后 |
|---|---|---|
| load + pool | 2106 MiB | 2106 MiB |
| round 0 | 5246 MiB | 2114 MiB |
| round 1 | 8338 MiB | 2114 MiB |
| round 2 | 10831 MiB | 2114 MiB |
| round 3 | **OOM** | 2115 MiB |

每轮泄漏约 3GB，三轮必 OOM。上游 HTTP 服务同样受影响（只是单请求泄漏慢）。

### 3. prefill 只算最后位置 logits

`(B,S,V)` → `(B,1,V)`。S=256 / V=151936 / bf16 时 **622MB → 2.4MB**。
采样只需要每个请求的最后一个位置，其余 255 个位置的 logits 纯属浪费。

### 4. 多 EOS

`generation_config.json` 的 `eos_token_id` 是 `[151645, 151643]`，
上游只认前者。漏掉 `<|endoftext|>` 会让请求空转到 `max_tokens`，
在 RL 里就是白白烧算力 + 产生垃圾 token 进入 loss。

---

## 尚未改动（有意保留）

- PagedAttention / CUDA Graph / 调度算法：`runner` 参数留了接口，M3 再开
- 流式 SSE（上游 v2 计划）
- 权重加载峰值显存：仍会在 GPU 上多复制一份 state_dict（约 1.2GB 瞬时），
  4GB 卡启动时 `reserved` 会到 2.9GB，之后回落；M2 若吃紧再优化
- `generate.py`（单请求脚本）的采样仍只返回 token，不带 logprob
