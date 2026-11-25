# 实验定位：受控 Tool-Calling Agentic RL

> 日期：2026-11-25
> 项目：HowToMove ｜ 模型：Qwen3-0.6B + LoRA
> 相关代码：`agent/`、`scripts/sft_agent.py`、`scripts/train_grpo_agent.py`

---

## 1. 一句话定位

> 本项目研究**受控迷宫环境中的 tool-calling Agentic RL**：
> LLM 通过多轮 `move` 工具调用与环境交互，先用 BFS 轨迹做 SFT 冷启动，
> 再用环境奖励和统一 GRPO 优化多步策略。

更准确地说，这是一个：

```text
controlled
single-tool
multi-turn
environment-reward-driven
Agentic RL
```

不是“通用自主智能体”，也不是单纯的 tool-call SFT。

---

## 2. 为什么算 Agentic RL

当前实验具备 agentic RL 的核心闭环：

```text
system: 静态地图
user:   Begin the maze.

assistant: tool_call(move)
tool:      JSON observation

assistant: tool_call(move)
tool:      JSON observation

...
```

它满足以下条件：

### 2.1 多步决策

动作会改变环境状态：

```text
position -> next position
```

后续 observation 和 reward 都依赖前一步动作，不是一次性的分类或问答。

### 2.2 工具调用是动作接口

动作空间不是隐藏的离散标签，而是模型生成的 tool call：

```text
move(up | down | left | right)
```

模型必须通过工具与环境交互，tool result 是下一步 observation。

### 2.3 环境奖励驱动

奖励来自环境规则，不是人类偏好模型：

```text
-0.1        步数惩罚
+1.0 × distance_progress
-2.0        撞墙/撞边界
+50.0       到达 goal
-10.0       非法 tool call
```

### 2.4 RL 优化轨迹回报

最终阶段使用 GRPO：

```text
R_i = sum(step_rewards in trajectory i)
A_i = (R_i - group_mean) / group_std
```

策略优化的是一条多步轨迹的累计环境回报，而不是单个 token 的监督标签。

### 2.5 SFT 只是冷启动

SFT 阶段主要学习：

- tool-call 格式；
- 与地图/坐标对应的基本移动；
- BFS 最短路径行为；
- 碰撞恢复样本。

RL 阶段才优化：

- 撞墙后的恢复；
- 更长的多步路径；
- hard map 上的成功率和回报。

---

## 3. 形式化描述

### 3.1 状态和 observation

```text
observation_t =
  system(map, start, goal, walls, tool contract)
  + interaction history
  + latest tool result
```

当前环境是**完全可观测、确定性的**：

- 地图完整放在 system prompt；
- 起点、终点、墙都可见；
- tool result 返回当前位置、是否撞墙、goal delta；
- 环境转移由规则决定。

### 3.2 动作

```text
a_t = move(direction)
direction ∈ {up, down, left, right}
```

非法 tool call 会导致 episode 结束。

### 3.3 奖励和轨迹

```text
τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)
R(τ) = Σ_t r_t
max_steps = 30（主实验）
```

### 3.4 优化目标

```text
max_θ  E_{τ ~ π_θ}[R(τ)]
```

训练使用 group-relative advantage，把同一条 hard map 上的 G 条轨迹互相比较。

---

## 4. 训练流程

```text
1. map-only 数据集
        ↓
2. BFS 最短路径 / 碰撞恢复轨迹
        ↓
3. SFT 冷启动
   (只监督 assistant tool_call + EOS)
        ↓
4. greedy hard-map screening
        ↓
5. 统一 GRPO（324 hard maps）
        ↓
6. held-out L0–L4 评估
        ↓
7. Maze Playground UI
```

关键点：

```text
数据只存地图；
SFT/GRPO 轨迹在训练时动态生成；
SFT 学格式和基本行为；
GRPO 用环境回报改进多步策略。
```

---

## 5. 和普通 SFT、RLHF 的区别

| 维度 | 普通 SFT | RLHF | 本实验 |
|---|---|---|---|
| 监督信号 | 专家 token | 人类偏好 | 环境 reward |
| 交互 | 单轮或静态 | 通常是回答级 | 多轮 tool call |
| 动作 | token | 文本回答 | `move` 工具调用 |
| 优化对象 | next-token | 偏好奖励 | trajectory return |
| 环境 | 无 | 无显式环境 | 确定性迷宫 |
| 恢复能力 | 依赖示范 | 间接 | GRPO 直接优化 |

所以更准确的称呼是：

```text
environment-driven, tool-calling Agentic RL
```

而不是：

```text
RLHF
通用 Agentic Intelligence
```

---

## 6. 当前结果支持什么

最终 GRPO checkpoint：

```text
/tmp/opencode/grpo_unified_chunk17/grpo_agent.pt
```

L0–L4：

```text
SFT  final: val 458/500, test 454/500
GRPO final: val 478/500, test 478/500
```

关键提升：

```text
L4: 80/79 -> 89/89
L3: 90/85 -> 97/95
L2: 91/92 -> 95/97
```

碰撞次数：

```text
SFT:  val 1111 / test 1190
GRPO: val  367 / test  343
```

泛化测试：

```text
8x8–16x16 零样本大小泛化基本可用
20x20 依赖 max_steps 预算
同一组墙换 start/goal：L4 68/75 = 90.7%
```

这些结果支持：

```text
Agentic RL 能改善多步 tool-calling 策略和 hard map 恢复能力
```

详情见：

```text
reports/GRPO/UNIFIED_FINAL.md
reports/GENERALIZATION.md
```

---

## 7. 当前限制

- 只有一个工具 `move`；
- 环境完全可观测、确定性；
- 动作空间固定为四个方向；
- 轨迹级 advantage 广播到每个 turn，不是严格的逐步 credit；
- 模型是 0.6B；
- 训练环境仍是固定 12×12；
- OOD 障碍物形状/密度尚未系统测试。

---
