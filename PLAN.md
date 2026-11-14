# AgenticRL 实施计划

> 用 GRPO 训练 Qwen3-0.6B，在 12×12 网格迷宫中通过工具调用完成长程寻路。
> 版本：v1（2026-11-14）

---

## 1. 目标与验收

| 项 | 指标 |
|---|---|
| 随机地图（含 L/U 墙）成功率 | ≥ 80% |
| 路径长度 / BFS 最短路 | ≤ 1.15 |
| 工具调用格式合法率 | ≥ 99% |
| 训练全程 | 4GB 卡无 OOM |

---

## 2. 决策记录

| 项 | 决定 | 理由 |
|---|---|---|
| 底座 | `Qwen3-0.6B`（`E:\codeall\Qwen3-0.6B`） | 28 层 / 1024 hidden / 8 KV heads / vocab 151936 / `tie_word_embeddings=true` |
| 引擎 | vendored fork → `AgenticRL\engine\` | 项目内置，引擎与训练一起改，改完仍是独立可用的推理项目 |
| 训练架构 | **路线 A：单进程 + 同一模型对象 + 手写循环** | 训推同源，`ratio` 第 0 步严格为 1；无权重同步；4GB 可行 |
| 训练框架 | 不用 TRL / verl | 手写 `Qwen3ForCausalLM` 接不进 HF `PreTrainedModel`；GRPO loss 直接照抄 TRL |
| 参数化 | `TrainAdapter` 抽象，`lora \| full` 切换 | 4GB 用 LoRA；全参为将来换卡预留，接口现在就要完整 |
| 上下文 | 无状态单轮 + 最近 3 步动作 | 4GB 吞吐；不撞引擎 `MAX_INPUT_LEN=2048` |
| 环境 | 自研 12×12，Minigrid 式 L/U 生成器 | Minigrid 的 reward / 动作 / 观测都不合用，只借生成器思路 |
| 推理采样 | `T=1.0, top_k=0, top_p=1.0` | 行为分布 = 训练重算分布 |

### 为什么不用现成训练框架

引擎是手写 PyTorch 模型（`qwen3/model.py`），不是 `transformers.PreTrainedModel`。
要保证「引擎产出的 logprob」和「训练时重算的 logprob」一致，最可靠的做法是**两边跑同一个模型对象**。
走 HF 路线需要额外一份 1.2GB 权重、两套实现的 bf16 数值差、以及锁死版本的私有 `rollout_func` 接口。
换到 24GB+ 卡时再评估是否切 TRL；`rollout` 的输出格式已按 TRL `rollout_func` 的 dict 设计，可平滑迁移。

---

## 3. 架构

```
┌──────────────────── 单进程 / 单 GPU ────────────────────┐
│  同一个 Qwen3ForCausalLM 实例（engine/ 的代码加载）       │
│                                                          │
│  rollout 阶段            │        train 阶段                │
│  ─ grad off              │        ─ adapter on              │
│  ─ Scheduler 连续批处理   │        ─ SDPA + grad ckpt        │
│  ─ T=1、无 top-k/p        │        ─ chunked CE              │
│  ─ 记录 raw logprob      │        ─ ref = adapter 关掉      │
└──────────────────────────────────────────────────────────┘
                            ↓
              optimizer.step() → 下一次 rollout 自动是新策略
```

---

## 4. 仓库结构

```
E:\codeall\AgenticRL\
├── PLAN.md
├── engine\                     fork（上游 ad04f0b，去掉 trace.json/logs/__pycache__/.git）
│   ├── FORK.md                 上游 commit + 改动清单
│   ├── engine_rl.py            【新增】批量 rollout Python API
│   ├── server.py               【改】logprob
│   ├── qwen3\request.py        【改】存 logprob
│   ├── qwen3\scheduler.py      【改】透传
│   └── ...（其余原样）
├── paths.py                    sys.path 注入 engine\
├── env\
│   ├── maze.py                 生成器 / BFS / step / reset
│   ├── render.py               紧凑网格渲染
│   ├── viz.py                  PNG / SVG / ANSI 导出（纯标准库）
│   ├── ui.py                   实时可视化 UI（HTTP + SSE + Canvas）
│   ├── test_maze.py            T1 + T5
│   ├── test_viz.py             渲染烟测
│   └── test_ui.py              UI 接口测试
├── rollout.py                  多轮 while、Policy 抽象、MockPolicy
├── dataset.py                  地图集读写 / 重建 / 采样 / 单步最优动作
├── data/                       数据集（4200 张图 + 300 评测状态 + manifest）
├── docs/DATASET.md             数据集生成指南
├── test_rollout.py             采集链路测试
├── test_dataset.py             数据集单测
├── grpo\
│   ├── adapter.py              TrainAdapter / LoRAAdapter / FullFTAdapter
│   ├── forward.py              SDPA + checkpoint + chunked CE
│   ├── loss.py                 GRPO loss
│   ├── batch.py                轨迹 → 训练张量
│   └── train.py
├── sft\coldstart.py
├── curriculum.py
├── eval.py
└── configs\base.py
```

**fork 方式**：复制文件树（保留 `from qwen3 import ...` 的包结构），`paths.py` 把 `engine\` 插到 `sys.path`，引擎内部 import 一行不改。上游 commit 写进 `FORK.md`。

---

## 5. 引擎 fork 改动清单（4 改 1 增）

| 文件 | 位置 | 改动 |
|---|---|---|
| `server.py` | `sample_one()` :138 | 返回 `(token, logprob)`；logprob 由**温度缩放前** `F.log_softmax(logits)` 取 |
| `qwen3/request.py` | `append()` :179 | 加 `logprob` 参数 + `output_logprobs: list[float]` |
| `qwen3/scheduler.py` | `on_step_done()` :93 | 透传 `next_logprobs` |
| `server.py` | `Engine._step()` / `Job` | 逐请求收 logprob，完成时写入 `job.logprobs` |
| `engine_rl.py` | 新增 | `RolloutEngine.generate_batch(prompt_ids, sampling) -> list[Sample]` |
| `server.py` `Engine._step` | 修 bug | `no_grad`（否则每批泄漏 3GB）+ prefill 只算最后位置 logits（622MB→2.4MB） |
| `qwen3/model.py` | 改 | `forward(last_positions=...)` |
| `qwen3/request.py` | 修 bug | 多 EOS（151645 + 151643） |

```python
@dataclass
class Sample:
    token_ids: list[int]
    logprobs: list[float]     # 与 token_ids 等长，fp32
    finish_reason: str        # stop | length
```

要点：

- 内部复用现成的 `Engine` / `Job` / `Scheduler`，不经过 HTTP。
- CUDA Graph 的 logits 是复用 buffer，logprob 必须当场 `.item()`。
- HTTP 接口保持可用，响应额外带 `token_ids` / `logprobs`（非标准字段）。
- 不碰 PagedAttention / CUDA Graph / 调度算法。

---

## 6. 环境设计

### 6.1 坐标与动作

`grid[y][x]`，y 向下。`up/down/left/right` → `y-1 / y+1 / x-1 / x+1`。
越界与撞墙都算碰撞。边界不设墙。

### 6.2 难度阶梯

| 级别 | 形态 | `dist - manhattan` 下限 |
|---|---|---|
| L0 | 空地 | `>= 8` 的最短距离，detour 恒为 0 |
| L1 | 竖墙 + 1 格门 | `>= 2` |
| L2 | L 型墙 | `>= 2` |
| L3 | U/C 型墙（窄门） | `>= 4` |
| L4 | 随机 1~3 个形状组合 | `>= 2` |

**detour 恒为偶数**：`L - manhattan = 2 × (逆向步数)`，所以 `2` = 至少走一步"远离"，
`4` = 至少走两步。`test_u_wall_forces_moving_away` 断言 L3/L4 的最短路**必然出现
曼哈顿距离增大的一步** —— 这正是「曼哈顿奖励会对抗 U 型墙」的判定式，也是选 BFS
距离做奖励的直接依据。

### 6.3 生成不变量

每次生成后必须满足，否则重采样（上限 200 次）：

1. 起点 ≠ 终点，且都不在墙上；
2. `bfs_dist(goal)[start] != INF`（可解）；
3. `bfs_dist(goal)[start] <= max_steps - 2`（否则 +50 永远拿不到）；
4. 级别约束（见 6.2 表）。

### 6.4 观测（约 110 token）

```
maze (12x12, #=wall .=free A=you G=goal, x=right y=down):
............
....#.......
....#..A....
....#.......
.........G..
...
pos=(5,2) goal=(9,4) delta=(+4,+2)
last=up,up,left
```

`delta` 是 M1 实测加进来的：Qwen3-0.6B 在 12×12 网格上无法稳定接地，
显式给出 `goal-pos` 后合法率从 4/8 提到 8/8（详见 `reports/M1.md` §6）。

**观测编码可切换**（`render_state(style=...)`，详见 `reports/obs_ablation.md`）：

| style | 形态 | prompt |
|---|---|---|
| `grid` | 12×12 ASCII（默认） | ~290 tok |
| `coords` | 坐标 + 障碍坐标列表 | ~287 tok |
| `rows` | 障碍按行分组、x 段压缩 | **~276 tok** |
| `bitmap` | 0/1 占用图 | ~405 tok |

消融结论：**四种编码输出完全一致**（30/30 全是 `right`），编码不是瓶颈。
只要提示里明说方向，模型能 100% 正确调用工具 —— 缺的是「delta → 方向」的映射，
这正是 SFT 的第一课。

### 6.5 实时可视化

零依赖本地服务：`python -m env.ui --open --demo` → http://127.0.0.1:7861/

- 浏览器里选级别/种子、WASD 手动走、播放最短路；
- 训练循环每步推一帧，页面经 SSE 实时刷新（不轮询）：

```python
ui = UIServer(port=7861).start()          # 训练进程内
ui.set_maze("L3", seed)
ui.show_step(maze, env.pos, {"action": a, "reward": r},
             llm={"text": raw_text, "logprob": lp, "tokens": n},
             prompt=prompt, episode="upd12/g3", turn=t)
```

- 页面「模型输出」面板显示模型**原始文本 + logprob + tokens + 延迟**，
  `<details>` 里是模型实际看到的 prompt —— 这是排查格式崩塌/复读机的第一现场。

- 跨进程用 `VizClient("http://127.0.0.1:7861").push(frame)`，失败静默、不拖慢训练。

### 6.6 奖励

```python
if not valid_tool_call:   -10, done=True    # 格式错，立即终止
elif collision_or_oob:     -2               # 坐标不变
elif reached_goal:        +50, done=True
else:                      bfs_d(s) - bfs_d(s') - 0.1
```

- `bfs_d` 是从终点反向 BFS 得到的全图距离场，O(144)，每张图算一次。
- 距离项等价于 Φ = −d、γ = 1 的 potential-based shaping，不改变最优策略。
- 与曼哈顿的唯一差别：**绕墙时不会先吃 −1**。
- 形式错只罚一次（立即终止），不是每步 −10。

---

## 7. Rollout 协议

```python
for t in range(max_steps):
    active = [k for k in trajs if not trajs[k].done]
    prompts = [render(trajs[k]) for k in active]
    samples = engine.generate_batch(prompts, sampling=T1)
    for k, s in zip(active, samples):
        action, ok = parse(s.token_ids)
        r, done = env.step(k, action)
        trajs[k].append(prompt_ids, s.token_ids, s.logprobs, r, ok)
```

- 每轮只发**当前仍存活**的轨迹，批量随轮数自然缩小。
- 轨迹记录 `{"prompt_ids", "completion_ids", "logprobs", "loss_mask"}`（TRL `rollout_func` 兼容格式）。
- Advantage 轨迹级算完再广播到每个 turn 的 completion token。
- KV 池按 `8 × 768 token = 320 块 ≈ 0.56GB` 自建，不用 server 的 2048 预算。

---

## 8. 训练侧

### 8.1 Adapter 抽象

```python
class TrainAdapter:
    def trainable_parameters(self) -> Iterable[nn.Parameter]
    def param_groups(self, base_lr) -> list[dict]
    def ref_context(self) -> ContextManager[Callable]
    def state_dict(self) / load_state_dict(sd)
```

| | `LoRAAdapter` | `FullFTAdapter` |
|---|---|---|
| 可训练 | `A/B` fp32，base 冻结 | 全部 + fp32 master |
| 目标 | `q/k/v/o_proj, gate/up/down_proj` | — |
| ref | `LORA_DISABLED=True`，同模型 | bf16 冻结快照 |
| 保存 | 只存 adapter | 全量 state_dict |
| lr / wd | 1e-5 / 0 | 2e-6 / 0.01 |

`FullFTAdapter` 必须 `lm_head.weight = embed_tokens.weight`：Qwen3-0.6B 是 tied，但引擎把它们建成了两个独立参数（`weights.py:67` 的 `strict=False` 掩盖了这点）。

### 8.2 forward 补丁（训练侧）

1. `attention.py:_standard_attention` → `F.scaled_dot_product_attention(is_causal=True)`；
   否则物化 `[B,H,S,S]`，S=2048/B=8 时约 2.1GB。
2. `lm_head` 分块（chunk 256），禁止全序列 vocab 矩阵。
3. `Qwen3DecoderLayer` 套 `torch.utils.checkpoint(use_reentrant=False)`。

### 8.3 Loss

```python
ratio = exp(new_logps - old_logps)              # old: fp32, detach
pg    = -min(ratio*A, clamp(ratio, 1±0.2)*A)
kl    = ref_logps - new_logps                   # k1
loss  = ((pg + beta*kl) * mask).sum() / mask.sum()   # token-level 归一化
```

### 8.4 超参

`G=8, max_steps=30, clip=0.2, beta_kl=0.01, S<=512, micro_bs=1, grad_accum=8`

### 8.5 显存

| | 4GB（LoRA） | 24GB（全参，预留） |
|---|---|---|
| 权重 master | 1.2G bf16 | 2.4G fp32 |
| 梯度 | — | 2.4G |
| Adam m/v | 0.04G | 4.8G |
| ref | 0（关 adapter） | 1.2G bf16 |
| KV 池 | 0.56G | 0.56G |
| 训练态 | 0.3G | 0.6G |
| **合计** | **≈2.5G** | **≈12G** |

---

## 9. 里程碑

| | 交付 | 通过条件 |
|---|---|---|
| **M0** | `env/` + 测试 | T1、T5 通过（纯 CPU） |
| **M1** | fork + logprob + `engine_rl.py` | T2 通过 |
| **M2** | `adapter.py` 双模式 + forward 补丁 + SFT | T6 > 95%，T8 通过 |
| **M3** | rollout + GRPO 3 步端到端 | T3、T4 通过 |
| **M4** | 30 步 + 课程控制器 | 成功率稳定 30~50%，T7 通过 |
| **M5** | 评估 + 报告 | 达到第 1 节指标 |

**T4（单图过拟合）是总闸门**：固定空地、20 次更新成功率不到 100%，不许进 M4。

---

## 10. 测试矩阵

| # | 测试 | 门槛 |
|---|---|---|
| T1 | 1000 张随机图（各级别 200 张）可解性 | 100% 可解且 `shortest <= max_steps-2` |
| T2 | 引擎 logprob vs 训练 forward 重算（20 条固定序列） | `max|Δ| < 1e-2` |
| T3 | 第 0 步 ratio | ∈ [0.99, 1.01] |
| T4 | 单图过拟合 20 步 | 成功率 100% |
| T5 | reward 单测（撞墙/越界/绕墙/到达/格式错/截断） | 逐项断言 |
| T6 | SFT 后格式合法率 | > 95% |
| T7 | 30 步轨迹无 `max_steps` 截断 | 到达率 > 0 |
| T8 | dummy 4 层模型双 adapter（CPU） | 参数计数 / tie / ref_context / save-load 往返 |

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| 0.6B 学不会 30 步规划 | 先空地，再加墙；不行降到 8×8 |
| 组内退化（全成功 / 全失败） | 难度控制器（成功率 EMA 目标 0.3~0.5）+ 退化组跳过 |
| 格式崩塌成复读机 | KL 从第一个更新就开 + SFT anchor |
| 4GB OOM | `micro_bs=1`、`S<=512`、KV 池 0.56G |
| 吞吐约 2 分钟/组 | 接受；早期 `max_steps=5` 短回合 |
| 训推数值漂移 | T2 硬门禁，不通过不训练 |

---

## 12. 非目标

不做多轮 transcript 上下文、不做分布式、不做 `enable_thinking`、
不改引擎调度 / attention 算法、不在 4GB 上跑全参。

---

## 13. 执行顺序

1. **M0**（纯 CPU）：`env/maze.py` → `env/render.py` → T1/T5
2. **M1**：fork 引擎 → logprob 补丁 → `engine_rl.py` → T2
3. **M2**：`adapter.py` → forward 补丁 → T8 → SFT → T6
4. **M3**：`rollout.py` → `grpo/` → T3 → T4
5. **M4**：`curriculum.py` → 30 步长程 → T7
6. **M5**：`eval.py` → 指标报告
