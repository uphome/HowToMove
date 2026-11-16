# FORK.md — 引擎核心导入说明

本目录是 `/data/hjt1/cppproject/LLMcpp` **main 分支的核心部分**，
只保留推理/训练真正需要的 `qwen3/` 包，不带 `server.py`、`main.py`、`bench*.py`、
`generate.py`、`test/` 等外围脚本。

| | |
|---|---|
| 上游仓库 | `/data/hjt1/cppproject/LLMcpp` |
| 上游分支 | `main` |
| 上游 commit | `21dae42`（`fix: 补齐推理路径缺失的 torch.no_grad`） |
| 导入时间 | 2026-11-16 |
| 导入方式 | `git archive HEAD qwen3`，只取已提交内容；工作区未提交的注释改动未带入 |

导入后 `paths.py` 把本目录插入 `sys.path`，因此引擎内部
`from qwen3 import ...` 不用改；项目侧通过 `engine_rl.py` 使用。

---

## 目录内容

```
engine/
├── qwen3/               上游核心（模型 / PagedKV / Scheduler / sampling / Triton kernels）
├── engine_rl.py         项目新增：RL 批量推理接口 RolloutEngine
├── FORK.md              本文件
└── .gitignore           上游自带
```

已经删除的上游外围文件：`server.py`、`main.py`、`generate.py`、
`batched_generate.py`、`bench.py`、`bench_batched.py`、`profile_decode.py`、
`chat_template.py`、`README.md`、`API.md`、`PERFORMANCE.md`、
`ENGINE_COMPARISON.md`。训练侧不需要 HTTP 服务或基准脚本。

---

## 对核心做的改动

### 1. `qwen3/model.py`

- **`last_positions`**：`Qwen3ForCausalLM.forward()` 支持只对每个请求的
  指定位置做 `lm_head`。推理只需要最后一个位置，prefill 的
  `(B,S,V)` 降到 `(B,1,V)`（S=256 / V=151936 / bf16 时 622MB → 2.4MB）。
- **`grad_checkpoint`**：`Qwen3Model` 增加训练开关，逐层
  `torch.utils.checkpoint(use_reentrant=False)`，把 B=8 的激活从 GB 级
  压到百 MB 级；推理默认关闭。

### 2. `qwen3/request.py`

- `eos_token_id` 支持 `int | tuple[int, ...]`，同时识别
  `<|im_end|>`（151645）和 `<|endoftext|>`（151643）。
- 新增 `output_logprobs`，`append(token, logprob)` 逐 token 记录
  **采样变换之前**的原始分布 logprob。

### 3. `qwen3/scheduler.py`

- `on_step_done(batch, next_tokens, next_logprobs=None)` 把每步 logprob
  透传到 `Request.append()`，供 GRPO 的 `old_logps` 使用。

### 4. `qwen3/kernels/paged_attention.py`

- **修复 fused qkv 的 stride bug**：`qkv_proj` 的 `k`/`v` 是融合张量的切片，
  `k_new`/`v_new` 的 batch 维 stride 不等于 `num_kv_heads * head_dim`。
  旧 kernel 按连续布局读取，导致 batch 里只有第一条请求正确、其余请求
  拿到 b=0 的 K/V（表现为批量 rollout 输出乱码）。现在 `update_kv_batch`
  显式传入 `k_new/v_new` 的 stride。

### 5. `engine_rl.py`（项目新增）

- 不经过 HTTP，直接用 `Scheduler` + `KVCachePool` 连续批处理；
- `generate_batch()` 同步等待整组完成，返回 `token_ids + logprobs`；
- top_k=0 / top_p=1 时用 `torch.multinomial` 做全词表精确采样，
  避免对 15 万词表做 full topk；
- KV 池按 RL 场景配置（短 prompt、G 路并发、`max_seq_len=768`）。

---

## 设计约束（不要改回去）

1. **logprob 必须取原始分布**：训练侧重算的是 `log_softmax(raw_logits)`。
   如果记录温度/top-k 之后的分布，GRPO 第 0 步 ratio 就不等于 1。
2. **推理必须 `no_grad`**：上游已在 `Engine`/图捕获路径补齐；本项目的
   `RolloutEngine.generate_batch` 也整体包了 `torch.no_grad()`。
3. **prefill 只算最后位置**：这是 4GB 卡能跑长 prompt 的关键；
   训练侧走 `grpo/forward.py` 的分块 `lm_head`，不依赖这个参数。
4. **多 EOS**：漏掉 `<|endoftext|>` 会让请求空转到 `max_tokens`，
   生成垃圾 token 直接进入 loss。

---

## 训练侧适配

上游 main 把 q/k/v 融合成 `qkv_proj`、gate/up 融合成 `gate_up_proj`，
因此 `grpo/adapter.py` 的 LoRA 目标改为：

```python
DEFAULT_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
```

`FullFTAdapter.tie_embeddings()` 不受影响（Qwen3-0.6B 仍是 tied embedding）。

---

## 已验证

- **批量 decode**：1/3/8 条不同长度 prompt 的 batch，全部输出合法 `move`
  工具调用（修复 stride bug 后）；
- `python -m pytest -q`：73 个纯 CPU 环境/轨迹/数据集测试通过；
- `scripts/calibrate_logprob.py --level L0 --seed 0`：T2 PASS；
- `scripts/train_phase1.py` 1 次更新冒烟：LoRA（fused targets）+ 梯度检查点
  + GRPO loss 全链路可跑。

> T2 的严格阈值（max|Δ|<1e-2）对“方向 token 接近并列”的探针很敏感：
> L1/L2/L3/L4 的部分 greedy 探针会出现 0.02–0.08 的 logprob 差，
> 旧 fork 在同探针上也有同量级差异。这是 bf16 下 paged decode 与
> 全序列 SDPA 的数值路径差异，不是 stride bug 回归；GRPO clip=0.2
> 可以覆盖。若要继续收紧，需要让训练/推理走同一套 attention 数值路径。

> 运行环境：`conda activate minimind`（Python 3.10 / torch 2.2.2+cu118 /
> transformers 4.57.6 / triton 2.2.0）。Triton 需要时设置
> `TRITON_PTXAS_PATH=/data/hjt1/anaconda3/envs/cuda_learn/bin/ptxas`。
