<div align="center">

<h1>HowToMove</h1>

<p><strong>受控环境下的 Tool-Calling Agentic RL</strong></p>

<p>
  <code>Qwen3-0.6B</code> ·
  <code>SFT 冷启动</code> ·
  <code>统一 GRPO</code> ·
  <code>多步迷宫导航</code>
</p>

<a href="reports/curves/rollouts/grpo_final_L4.gif">
  <img src="reports/curves/rollouts/grpo_final_L4.gif"
       alt="L4 success case" width="520">
</a>

<p><sub>L4 hard-wall 成功案例 · 12 steps · 0 collisions</sub></p>

</div>

---

## 项目定义

HowToMove 是一个**受控环境下的 tool-calling Agentic RL** 项目，研究语言模型
能否通过多轮工具调用完成多步导航，并利用环境反馈从撞墙和错误中恢复。

| 维度 | 定义 |
|---|---|
| 环境 | 静态迷宫：完整地图、墙体、start、goal |
| 策略 | Qwen3-0.6B + LoRA |
| 动作 | assistant 生成 `move(direction)` tool call |
| 观察 | tool result JSON：位置、是否撞墙、goal delta |
| 奖励 | 环境规则：步数、距离进展、碰撞、到达目标 |
| 训练 | BFS 轨迹 SFT 冷启动 → hard map 筛选 → 统一 GRPO |
| 评估 | held-out 地图、墙体布局、尺寸、start/goal |

核心问题：

1. SFT 冷启动能否让模型学会 tool protocol 和基本导航；
2. GRPO 能否用环境奖励改善多步策略、减少撞墙循环；
3. 学到的策略能否泛化到未见过的墙体、尺寸和 start/goal。

边界：

> 这是**受控、单工具、确定性环境**下的 Agentic RL，不是通用自主智能体，
> 不是 RLHF，也不是单纯的 tool-call SFT。

完整定义、形式化描述、训练流程和结论边界见：

```text
docs/POSITIONING.md
```

---

## 1. 目标

最终目标是处理：

- 任意 12×12 地图；
- 任意墙体布局；
- 任意 start / goal；
- 多步轨迹；
- 撞墙反馈与恢复；
- 统一 SFT + GRPO 训练。

当前版本使用固定 12×12、`max_steps=30`。

Agent 协议：

```text
system: 静态地图
user:   Begin the maze.

assistant: <tool_call>move(direction)</tool_call>
tool:      compact JSON result

assistant: ...
...
```

---

## 2. 数据集制作

### 2.1 数据单位

当前只保存 **Map**，不保存轨迹：

```text
Map = 尺寸 + 具体墙体坐标 + start + goal
```

```json
{
  "id": "L1-0123456789",
  "level": "L1",
  "width": 12,
  "height": 12,
  "max_steps": 30,
  "walls": [[0,4], [1,4], [2,4]],
  "start": [4,7],
  "goal": [2,2],
  "shortest": 9,
  "detour": 2,
  "hash": "..."
}
```

Map 唯一性由以下内容哈希决定：

```text
width + height + sorted absolute walls + start + goal
```

同一墙形放在不同位置，或 start/goal 不同，都视为不同 map。

### 2.2 数据规模

```text
data/maps/train.jsonl   4000
data/maps/val.jsonl      500
data/maps/test.jsonl     500
```

每个 level：

| level | train | val | test |
|---|---:|---:|---:|
| L0 | 800 | 100 | 100 |
| L1 | 800 | 100 | 100 |
| L2 | 800 | 100 | 100 |
| L3 | 800 | 100 | 100 |
| L4 | 800 | 100 | 100 |

### 2.3 Split

- L1–L4：按**具体墙体集合** `wall_set` 隔离；
- L0：只有一个空墙集合，按 map/start-goal 划分。

同一个 `wall_set` 不会跨 train/val/test（L0 除外）。

### 2.4 生成与检查

```bash
python scripts/build_maps_v3.py \
  --out data \
  --seed 7 \
  --per-level 1000 \
  --train 800 --val 100 --test 100

python scripts/check_maps_v3.py --data data
```

更多说明见：

```text
reports/DATASET.md
```

---

## 3. Agent 设置

### 3.1 模型

```text
Qwen3-0.6B
LoRA: r=16, alpha=32
```

LoRA 权重可按任务加载；基础模型保持冻结。
> 0.6B的模型 对于这类任务会不会太小了。
### 3.2 工具

只有一个工具：

```python
{
    "name": "move",
    "arguments": {
        "direction": "up | down | left | right"
    }
}
```

### 3.3 静态地图 system prompt

由 `agent/runtime.py::build_agent_system` 生成：

```text
You are an agent solving a 12x12 maze.
Coordinates: x increases to the right, y increases downward.
Start: (4, 7)
Goal: (2, 2)
Walls: (0,4) (1,4) (2,4) ...

You control one cell. Each turn, call the move tool with exactly one
direction: up, down, left, or right.

The move tool returns a compact JSON result:
- moved
- ok
- pos
- blocked
- blocked_by
- done
- goal_delta
- reason

If you hit a wall or the map boundary, your position does not change.
If done=true, stop calling move.
Always call move. Never reply in plain text.
```

初始 user 只有：

```text
Begin the maze.
```

### 3.4 Tool result

正常移动：

```json
{
  "moved": "up",
  "ok": true,
  "pos": [4, 6],
  "blocked": false,
  "blocked_by": null,
  "done": false,
  "goal_delta": [2, -4],
  "reason": "You moved up."
}
```

撞墙：

```json
{
  "moved": "up",
  "ok": false,
  "pos": [4, 6],
  "blocked": true,
  "blocked_by": "wall",
  "done": false,
  "goal_delta": [2, -4],
  "reason": "You hit a wall. Your position did not change."
}
```

### 3.5 History

- history window：12 对 assistant/tool；
- 默认关闭显式 thinking chain；
- 每一步保留 assistant tool call 和 tool result。

---

## 4. LLM Prompt 和 SFT 数据

SFT 轨迹由 BFS 最短路径动态生成，不保存到 `data/`：

```text
Map → BFS shortest path → assistant/tool turns
```

训练目标只有：

```text
assistant tool_call + EOS
```

system、user、tool result 和 padding 全部 mask。

为了学习撞墙恢复，支持动态 recovery 样本：

```bash
--recovery-per-task N
--recovery-depth K
```

含义：

```text
在最优路径状态注入 blocked 动作
→ 生成 blocked=true 的 tool result
→ 从该状态继续训练后续 K 个正确动作
```

当前 L0–L4 主 SFT 使用：

```text
recovery_per_task=2
recovery_depth=1
```

L4 的更长 recovery 实验使用：

```text
recovery_per_task=2
recovery_depth=3
```

---

## 5. 训练

### 5.1 SFT

脚本：

```text
scripts/sft_agent.py
```

示例：

```bash
python scripts/sft_agent.py \
  --levels L4 \
  --maps 800 \
  --epochs 2 \
  --recovery-per-task 2 \
  --init-adapter /tmp/opencode/sft_agent_v3_l3/sft_agent.pt \
  --batch-size 8 \
  --lr 5e-5 \
  --history-window 12 \
  --max-seq-len 4096 \
  --val-limit 100 \
  --test-limit 100 \
  --eval-batch-size 8 \
  --seed 7
```

SFT 结果（最终 L4 checkpoint）：

| level | val | test |
|---|---:|---:|
| L0 | 100/100 | 100/100 |
| L1 | 97/100 | 98/100 |
| L2 | 91/100 | 92/100 |
| L3 | 90/100 | 85/100 |
| L4 | 80/100 | 79/100 |

#### SFT Loss 曲线

L0–L4 顺序 SFT 的逐 step loss（浅蓝）与平滑曲线（深蓝）。
每个 level 从上一级 checkpoint 初始化；L4 使用主实验的 `recovery_depth=1`。

| L0 | L1 |
|:---:|:---:|
| [![L0 SFT loss](reports/curves/sft/sft_agent_v3_l0_loss.png)](reports/curves/sft/sft_agent_v3_l0_loss.png) | [![L1 SFT loss](reports/curves/sft/sft_agent_v3_l1_loss.png)](reports/curves/sft/sft_agent_v3_l1_loss.png) |
| **L2** | **L3** |
| [![L2 SFT loss](reports/curves/sft/sft_agent_v3_l2_loss.png)](reports/curves/sft/sft_agent_v3_l2_loss.png) | [![L3 SFT loss](reports/curves/sft/sft_agent_v3_l3_loss.png)](reports/curves/sft/sft_agent_v3_l3_loss.png) |
| **L4** | |
| [![L4 SFT loss](reports/curves/sft/sft_agent_v3_l4_loss.png)](reports/curves/sft/sft_agent_v3_l4_loss.png) | |

> **为什么 epoch 切换处 loss 会明显上升？**
>
> SFT 按**精确 prompt 长度**分桶训练：同一 batch 内 prompt 长度一致，
> bucket 内随机 shuffle，bucket 顺序按短 → 长。每个 epoch 结束时最后一批
> 是长 prompt，下一个 epoch 又从最短 bucket 开始，因此 epoch 边界会出现
> 一次明显的 loss 抬升。这是**数据分布和 bucket 顺序重置**，不是训练发散。
> 以 L0 为例：epoch 1 最后 10 step 平均 loss 约 `0.000003`，epoch 2 开头
> 10 step 约 `0.0423`；但 epoch 1 平均 loss 为 `0.0263`，epoch 2 为
> `0.0218`，整体仍然在下降。若希望边界更连续，可以后续尝试每个 epoch
> 随机打乱 bucket 顺序。详见
> [`reports/SFT/L0.md`](reports/SFT/L0.md) 的 3.1 节。

L4 的 `recovery_depth=3` 对比曲线见：
[`reports/SFT/L4_RECOVERY_DEPTH3.md`](reports/SFT/L4_RECOVERY_DEPTH3.md)。

详细结果：

```text
reports/SFT/SUMMARY.md
reports/SFT/L0.md ... reports/SFT/L4.md
```

L4 的更长 recovery 实验：

```text
recovery_per_task=2
recovery_depth=3
```

结果：

```text
L4 val/test: 79/81
L3 val/test: 89/89
L4 collisions: 401/395
```

相比 L4 SFT，L4 test +2、L3 test +4，但 L1/L2 有轻微回退。
详见：

```text
reports/SFT/L4_RECOVERY_DEPTH3.md
```

### 5.2 Baseline

base Qwen3 在 L0/L1：

```text
success = 0/100
format errors = 100/100
```

详见：

```text
reports/BASELINE.md
```

### 5.3 GRPO

GRPO 使用最终 L4 SFT checkpoint，采用 hard map pool：

```text
data/grpo/l1_l4_pool.json
```

筛选：

```text
3200 screened maps
2876 easy
324 hard
```

GRPO 配置：

```text
G:            8
temperature:  0.8
top_p:        0.95
lr:           1e-5
beta:         0
history:      12
max_seq_len:  4096
```

已完成：

```text
324 / 324 hard maps
17 chunks
```

最终 checkpoint：

```text
/tmp/opencode/grpo_unified_chunk17/grpo_agent.pt
```

最终 L0–L4 评估：

| level | SFT val/test | chunk1 val/test | final GRPO val/test |
|---|---:|---:|---:|
| L0 | 100/100 | 100/100 | 99/100 |
| L1 | 97/98 | 98/97 | 98/97 |
| L2 | 91/92 | 92/91 | **95/97** |
| L3 | 90/85 | 90/87 | **97/95** |
| L4 | 80/79 | 80/79 | **89/89** |
| **总计** | **458/454** | **460/454** | **478/478** |

```text
最终 GRPO: val 478/500 = 95.6%
           test 478/500 = 95.6%
```

相比 SFT：

```text
val:  +20 个任务
test: +24 个任务
```

#### GRPO 曲线（MA 20）

浅色为原始逐 update 数据，深色为窗口 20 的 centered moving average。

| mean reward | success rate |
|:---:|:---:|
| [![GRPO final reward](reports/curves/grpo/grpo_unified_final_reward.png)](reports/curves/grpo/grpo_unified_final_reward.png) | [![GRPO final success](reports/curves/grpo/grpo_unified_final_success.png)](reports/curves/grpo/grpo_unified_final_success.png) |
| **loss** | **KL** |
| [![GRPO final loss](reports/curves/grpo/grpo_unified_final_loss.png)](reports/curves/grpo/grpo_unified_final_loss.png) | [![GRPO final KL](reports/curves/grpo/grpo_unified_final_kl.png)](reports/curves/grpo/grpo_unified_final_kl.png) |
| **reward range** | |
| [![GRPO final reward range](reports/curves/grpo/grpo_unified_final_reward_range.png)](reports/curves/grpo/grpo_unified_final_reward_range.png) | |

GRPO 详情：

```text
reports/GRPO/SCREENING.md
reports/GRPO/PILOT_L1.md
reports/GRPO/UNIFIED_CHUNK1.md
reports/GRPO/UNIFIED_FINAL.md
```

### 5.4 泛化性测试

使用最终 GRPO checkpoint 做零样本测试，只生成评估集，不重新训练。

固定 `max_steps=30`：

| size | L0 | L4 | total |
|---|---:|---:|---:|
| 8×8 | 100/100 | 86/100 | 186/200 |
| 10×10 | 100/100 | 87/100 | 187/200 |
| 12×12 | 99/100 | 84/100 | 183/200 |
| 16×16 | 97/100 | 84/100 | 181/200 |
| 20×20 | 93/100 | 77/100 | 170/200 |

> 12×12 是重新采样的对照；原始标准 test 的 L4 是 89/100，差异来自不同 held-out 地图。

延长步数预算：

| size | max_steps | L0 | L4 | total |
|---|---:|---:|---:|---:|
| 16×16 | 64 | 100/100 | 93/100 | 193/200 |
| 20×20 | 80 | 98/100 | 93/100 | 191/200 |

> 固定 30 步和延长预算使用不同地图样本，不是严格的同图预算消融。

同墙不同 `start/goal`：

```text
L4: 68/75 = 90.7%
```

原始 L4 test 为 89/100，同墙换起点/终点后没有明显下降。

当前结论：

- 8×8–16×16 的零样本大小泛化基本可用；
- 20×20 固定 30 步会明显受预算限制，延长预算后恢复；
- 同一组墙换 `start/goal` 基本保持；
- **尚未测试全新的障碍物形状、密度和复杂连通墙**。

详见：

```text
reports/GENERALIZATION.md
scripts/build_generalization_sets.py
```

### 5.5 Reward

每一步的 reward：

```text
-0.1        步数惩罚
+1.0 × distance_progress
-2.0        撞墙/撞边界
+50.0       到达 goal
-10.0       非法 tool call
```

当前重复访问惩罚关闭。

GRPO 使用轨迹级总 reward：

```text
R = sum(step_rewards)
A_i = (R_i - group_mean) / group_std
```

同一条轨迹的所有 turn 共享该 advantage。

### 5.6 训练与推理框架

#### 训练

```text
base model:   Qwen3-0.6B
adapter:      LoRA, r=16, alpha=32
framework:    PyTorch + 自定义训练循环
```

训练没有使用 HuggingFace Trainer / TRL：

```text
scripts/sft_agent.py          SFT 冷启动
scripts/train_grpo_agent.py   多步 GRPO
agent/                        tool-calling agent runtime
grpo/                         LoRA / GRPO / multi-step trainer
```

SFT 只监督 assistant tool call + EOS；GRPO 使用完整轨迹的环境 reward
和 group-relative advantage。

#### 推理

项目使用手搓的 Qwen3 推理引擎
（[Qwen3-engine-from-scratch](https://github.com/uphome/Qwen3-engine-from-scratch)）：

```text
engine/                       推理核心
engine/engine_rl.py           RolloutEngine：批量生成 + logprob
agent/policy.py               ChatPolicy：messages → Qwen3 生成
agent/runtime.py              MazeAgent：多步 tool-call loop
```

推理引擎包含 paged KV cache、scheduler、批量生成和 Triton kernels；
同一套推理链路用于：

```text
SFT 评估
GRPO rollout
最终 held-out 评估
Maze Playground UI
```

推理引擎来源与致谢见文末。

---

## 6. 结果可视化

最终模型的 L4 成功轨迹见 README 开头的 **效果预览**。

轨迹说明：

```text
map:        L4-001ed3d5ea
level:      L4
steps:      12
collisions: 0
```

蓝色线表示实际执行路径。更早的 chunk1 可视化保留在历史报告中：

```text
reports/GRPO/UNIFIED_CHUNK1.md
```

### 数据集地图

```text
reports/curves/maps/maps_L1_L4_overview.png
reports/curves/maps/maps_L1.png
reports/curves/maps/maps_L2.png
reports/curves/maps/maps_L3.png
reports/curves/maps/maps_L4.png
```

---

## 7. Maze Playground UI

新增独立页面，可以自由编辑地图并让模型实时运行：

```text
http://127.0.0.1:7861/playground
```

![Maze Playground UI](reports/curves/ui/maze_playground.jpg)

启动：

```bash
python -m env.ui \
  --playground-adapter /tmp/opencode/grpo_unified_chunk17/grpo_agent.pt \
  --open
```

功能：

- 自由设置 `width` / `height`；
- 画墙、擦墙；
- 自由设置 start / goal；
- 导入/导出地图 JSON；
- 设置 model、adapter、temperature、top_p、max_tokens、
  history_window、max_seq_len、seed；
- 运行/停止模型，实时查看 tool call、位置、reward、碰撞和轨迹；
- 自动校验可达性、最短路径和 `max_steps`。

adapter 留空时使用 base model。模型第一次运行时才加载，也可以手动
「加载模型 / 释放模型」。

详细说明：

```text
docs/PLAYGROUND.md
```

---

## 8. 目录结构

```text
agent/                 Agent runtime / policy / group
env/                   maze environment
grpo/                  GRPO adapter / loss / trainer
scripts/               data / SFT / GRPO / visualization
data/
├── maps/              map-only v3
├── manifest.json
└── grpo/              hard map pool
reports/
├── INDEX.md
├── DATASET.md
├── BASELINE.md
├── SFT/
├── GRPO/
└── curves/
```

完整报告索引：

```text
reports/INDEX.md
```

---

## 9. 当前不足

1. **数据集生成不够好，不够多样化**
   - 当前墙体主要来自竖墙、L 型墙、U 型墙的随机组合；
   - 障碍物位置和组合偏随机，缺少结构化难度分层的生成器；
   - 还需要更多复杂连通墙、窄走廊、房间和密度分层测试。

2. **模型太小**
   - 当前使用 Qwen3-0.6B，长轨迹规划和 OOD 泛化能力有限；
   - 更大模型、更多 multi-step 数据和多尺寸训练仍是后续方向。

3. **提示词没有实时周围环境**
   - 当前 system prompt 是完整静态地图；
   - 每轮主要依赖 tool result 和 history，没有显式的局部邻域/射线观测；
   - 这会让模型在 hard wall 附近缺少“眼前环境”的归纳偏置，容易形成局部撞墙循环。

4. **没有开启思考链推理**
   - 当前 `ChatPolicy` 默认 `enable_thinking=False`；
   - 模型直接生成 tool call，没有显式的计划、反思或逐步推理链；
   - 在 L4 hard wall 上，缺少内部 deliberation 可能是反复撞墙的原因之一；
   - 开启 thinking 会增加 token、延迟和训练成本，需要单独做训练与评估对比。

---

## 10. 致谢

- [uphome/Qwen3-engine-from-scratch](https://github.com/uphome/Qwen3-engine-from-scratch)：
  手搓的 Qwen3 推理引擎，本项目的 `engine/` 推理核心基于它整理和接入。
- Qwen3-0.6B：提供 base model。
- 项目内的 SFT、GRPO、Agent runtime、可视化和 Playground 代码均围绕该推理引擎构建。

---
