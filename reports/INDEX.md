# Reports Index

> 项目：静态地图 tool-calling Agent
> 最后更新：2026-11-23

---

## 当前状态

```text
数据：map-only v3，5000 maps
SFT：L0–L4 已完成，val 91.6% / test 90.8%
GRPO：324 / 324 hard maps 已完成
最终 GRPO：val 478/500 (95.6%) / test 478/500 (95.6%)
```

当前主要问题：

```text
L4 仍有少量失败，格式错误 2(val)+1(test)
但 L4 已从 SFT 的 80/79 提升到 89/89
```

---

## 1. 数据集

| 文档 | 说明 |
|---|---|
| [`DATASET.md`](DATASET.md) | map-only v3：生成、split、map 唯一性、校验、可视化 |

核心数据：

```text
data/maps/{train,val,test}.jsonl
data/manifest.json
data/grpo/l1_l4_pool.json
```

---

## 2. Baseline

| 文档 | 说明 |
|---|---|
| [`BASELINE.md`](BASELINE.md) | base Qwen3 在 L0/L1 的完整指标 |

结论：

```text
base L0/L1 success = 0
format errors = 100%
```

---

## 3. SFT

总览：

| 文档 | 说明 |
|---|---|
| [`SFT/SUMMARY.md`](SFT/SUMMARY.md) | 最终 L4 checkpoint 对 L0–L4 的汇总 |

单级：

```text
SFT/L0.md
SFT/L1.md
SFT/L2.md
SFT/L3.md
SFT/L4.md
SFT/L4_RECOVERY_DEPTH3.md
```

最终 L4 checkpoint：

| level | val | test |
|---|---:|---:|
| L0 | 100/100 | 100/100 |
| L1 | 97/100 | 98/100 |
| L2 | 91/100 | 92/100 |
| L3 | 90/100 | 85/100 |
| L4 | 80/100 | 79/100 |

---

## 4. GRPO

| 文档 | 说明 |
|---|---|
| [`GRPO/SCREENING.md`](GRPO/SCREENING.md) | L1–L4 greedy 筛选：3200 → 324 hard maps |
| [`GRPO/SIGNAL_L1.md`](GRPO/SIGNAL_L1.md) | L1 SFT 的 G=8 rollout 信号探测 |
| [`GRPO/PILOT_L1.md`](GRPO/PILOT_L1.md) | 早期 L1 GRPO pilot |
| [`GRPO/UNIFIED_CHUNK1.md`](GRPO/UNIFIED_CHUNK1.md) | 统一 GRPO chunk1：训练、reward/KL/loss、失败案例 |
| [`GRPO/UNIFIED_FINAL.md`](GRPO/UNIFIED_FINAL.md) | 统一 GRPO 最终结果：324/324、最终 L0–L4 评估、曲线 |

当前 pool：

```text
L1: 29
L2: 45
L3: 125
L4: 125
total: 324
```

训练覆盖：

```text
324 / 324
```

最终 checkpoint：

```text
/tmp/opencode/grpo_unified_chunk17/grpo_agent.pt
```

---

## 5. 图表

```text
curves/
├── maps/       数据集地图样例
├── sft/        L0–L4 SFT loss
├── grpo/       GRPO reward / KL / loss / success（chunk1 + final）
└── rollouts/   GRPO chunk1 推理轨迹和 GIF
```

---

## 6. 相关代码

```text
scripts/build_maps_v3.py
scripts/check_maps_v3.py
scripts/sft_agent.py
scripts/screen_grpo_maps.py
scripts/build_grpo_pool.py
scripts/train_grpo_agent.py
scripts/render_agent_rollouts.py
```
