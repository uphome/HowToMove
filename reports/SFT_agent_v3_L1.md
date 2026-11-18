# Agent SFT L1：map-only v3

> 日期：2026-11-18 ｜ 模型：Qwen3-0.6B + LoRA
> 数据：`data/maps/{train,val,test}.jsonl` v3
> 训练脚本：`scripts/sft_agent.py`

---

## 1. 结论

L1 从 L0 checkpoint 出发，继续训练 2 个 epoch：

| 模型 | split | success | mean steps | collisions | format errors |
|---|---|---:|---:|---:|---:|
| base | val | 0/100 | 10.1 | 58 | 100 |
| base | test | 0/100 | 10.1 | 52 | 100 |
| SFT | val | **75/100** | 17.4 | 765 | 0 |
| SFT | test | **78/100** | 17.0 | 685 | 0 |

第一轮单独 1 epoch：

```text
val:  67/100
test: 65/100
```

继续 1 epoch 后：

```text
val:  75/100
test: 78/100
```

L1 还没有达到 L0 的饱和状态，仍存在大量撞墙/超时失败。

---

## 2. 数据

```text
L1 train maps: 800
L1 val maps:   100
L1 test maps:  100
```

L1 当前数据按具体墙体集合隔离：

```text
L1 train: 800 个不同 wall sets
L1 val:   100 个不同 wall sets
L1 test:  100 个不同 wall sets
```

因此：

- 同一组具体 walls 不会跨 train/val/test；
- 当前每个 wall set 只生成一张 map；
- L1 的验证是未见过墙体集合上的泛化。

SFT 数据：

```text
turns: 10497
actions:
  up=2564, down=2548, left=2680, right=2705
```

---

## 3. 训练配置

```text
base:           Qwen3-0.6B
adapter:        LoRA, r=16, alpha=32
initialization: L0 checkpoint
lr:             5e-5
batch size:     8
epochs:         2（1 + 1）
history window: 12
max seq len:    4096
recovery/task:  2
seed:           7
total steps:    3366
```

Loss 曲线（两段训练拼接，浅色 per-step，粗线 50-step MA）：

![L1 v3 loss](curves/sft_agent_v3_l1_loss.png)

约 step 1683 附近是两次训练的边界，loss 因 bucket 顺序重置出现
抬升；整体仍是下降趋势。

---

## 4. 评估

使用与训练相同的 Agent 协议：

```text
system: 静态地图
user:   Begin the maze.
assistant: move(direction)
tool: compact JSON
```

greedy 解码，`history_window=12`，每批 8 个独立 agent。

动作分布：

```text
val:  up=296, down=416, left=467, right=566
test: up=342, down=346, left=365, right=648
```

四个方向都正常使用，format errors 为 0。

---

## 5. 与 baseline 对比

| 级别 | base success | SFT success | base format errors | SFT format errors |
|---|---:|---:|---:|---:|
| L0 val | 0/100 | 100/100 | 100 | 0 |
| L0 test | 0/100 | 100/100 | 100 | 0 |
| L1 val | 0/100 | 75/100 | 100 | 0 |
| L1 test | 0/100 | 78/100 | 100 | 0 |

GRPO 继续训练后的结果：

| 模型 | val | test |
|---|---:|---:|
| L1 SFT | 75/100 | 78/100 |
| L1 SFT + GRPO | **87/100** | **88/100** |

base model 完整指标见：

```text
reports/AGENT_BASELINE.md
```

GRPO 详情见：

```text
reports/GRPO_AGENT_v3_L1.md
```

---

## 6. 失败模式

主要失败仍然是：

- 撞墙后重复同一方向；
- 30 步耗尽；
- 部分新 wall set 上无法找到绕行路径。

L1 的 collisions 仍然很高：

```text
val collisions:  765
test collisions: 685
```

说明 recovery=2 只教会了部分撞墙恢复，还没有解决所有布局。

---

## 7. 复现

从 L0 checkpoint 训练 L1：

```bash
python scripts/sft_agent.py \
  --levels L1 \
  --maps 800 \
  --epochs 1 \
  --init-adapter /tmp/opencode/sft_agent_v3_l0/sft_agent.pt \
  --recovery-per-task 2 \
  --batch-size 8 \
  --lr 5e-5 \
  --history-window 12 \
  --max-seq-len 4096 \
  --val-limit 100 \
  --test-limit 100 \
  --eval-batch-size 8 \
  --seed 7 \
  --plot-dir /tmp/opencode/sft_agent_v3_l1/curves \
  --save-dir /tmp/opencode/sft_agent_v3_l1 \
  --json /tmp/opencode/sft_agent_v3_l1.json
```

继续训练第二个 epoch：

```bash
python scripts/sft_agent.py \
  --levels L1 \
  --maps 800 \
  --epochs 1 \
  --init-adapter /tmp/opencode/sft_agent_v3_l1/sft_agent.pt \
  --recovery-per-task 2 \
  --batch-size 8 \
  --lr 5e-5 \
  --history-window 12 \
  --max-seq-len 4096 \
  --val-limit 100 \
  --test-limit 100 \
  --eval-batch-size 8 \
  --seed 7 \
  --plot-dir /tmp/opencode/sft_agent_v3_l1_e2/curves \
  --save-dir /tmp/opencode/sft_agent_v3_l1_e2 \
  --json /tmp/opencode/sft_agent_v3_l1_e2.json
```
