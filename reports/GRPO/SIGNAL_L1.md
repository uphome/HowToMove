# L1 SFT 的 GRPO 信号探测

> 日期：2026-11-18 ｜ checkpoint：L1 v3 e2
> 数据：map-only v3，L1 train maps
> 脚本：`scripts/grpo_signal_agent.py`
>
> 后续 GRPO 训练结果见
> [`PILOT_L1.md`](PILOT_L1.md)。

---

## 1. 设置

```text
model:        Qwen3-0.6B + LoRA
checkpoint:   /tmp/opencode/sft_agent_v3_l1_e2/sft_agent.pt
maps:         20 张 L1 train map
G:            8
temperature:  0.8
top_p:        0.95
history:      12
max steps:    30
```

每张 map 独立采样 8 条 Agent 轨迹，按组统计 reward、success 和
collision 差异。

---

## 2. 结果

```text
mixed groups:    5 / 20  (25%)
all-fail groups: 0 / 20
all-success:     15 / 20
mean reward range: 31.14
```

没有出现“所有组都完全失败”的退化情况，说明 GRPO 至少存在一定
reward 差异。

### mixed 组

| map | success | reward range | mean reward | total collisions |
|---|---:|---:|---:|---:|
| L1-0148318833 | 6/8 | 116.0 | 29.45 | 54 |
| L1-01ed9941af | 7/8 | 84.7 | 39.15 | 61 |
| L1-03d07e8620 | 7/8 | 113.6 | 47.88 | 26 |
| L1-059e65c513 | 6/8 | 121.4 | 33.93 | 56 |
| L1-0629721cfb | 6/8 | 69.8 | 43.16 | 53 |

这些组同时包含成功轨迹和失败/超时轨迹，适合 GRPO 做组内相对优化。

### all-success 组

15/20 张 map 的 8 条轨迹都成功，但 reward 仍然会因为：

- 路径长度；
- 碰撞次数；
- revisit / step penalty；

产生一定差异。例如：

```text
all-success reward range 通常约 2–28
```

因此它们仍可提供“更短、更少碰撞”的信号，但无法直接学习从失败中
恢复。

---

## 3. 判断

GRPO 有信号，但不能只靠随机训练 map：

- mixed 组只有 25%；
- 大部分训练 map 已经被 SFT 学会；
- 反复撞墙的失败主要集中在少数 hard maps。

因此 GRPO 实验应该：

1. 优先选择 mixed rollout 的 map；
2. 或先做 hard-map / failure-state 采样；
3. 避免大量 all-success、零优势的组浪费更新。

---

## 4. 复现

```bash
python scripts/grpo_signal_agent.py \
  --init-adapter /tmp/opencode/sft_agent_v3_l1_e2/sft_agent.pt \
  --level L1 \
  --maps 20 \
  --g 8 \
  --temperature 0.8 \
  --top-p 0.95 \
  --out /tmp/opencode/grpo_signal_v3_l1.json
```

完整数据：

```text
/tmp/opencode/grpo_signal_v3_l1.json
```
