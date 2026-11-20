# Agent SFT 全部 level 汇总（map-only v3）

> 日期：2026-11-19 ｜ 模型：Qwen3-0.6B + LoRA
> 数据：`data/maps/{train,val,test}.jsonl`
> Split：L1–L4 按具体 wall set 隔离；L0 按 map/start-goal

---

## 1. 最终 L4 checkpoint 的全 level 结果

最终模型：

```text
/tmp/opencode/sft_agent_v3_l4/sft_agent.pt
```

完整 val/test 100：

| level | val | test | mean steps (val/test) | collisions (val/test) | format errors |
|---|---:|---:|---:|---:|---:|
| L0 | **100/100** | **100/100** | 11.6 / 11.9 | 7 / 1 | 0 / 0 |
| L1 | **97/100** | **98/100** | 12.5 / 12.4 | 85 / 64 | 0 / 0 |
| L2 | **91/100** | **92/100** | 14.4 / 14.3 | 223 / 221 | 0 / 0 |
| L3 | **90/100** | **85/100** | 15.1 / 16.6 | 291 / 433 | 0 / 0 |
| L4 | **80/100** | **79/100** | 17.1 / 17.5 | 505 / 471 | 0 / 0 |

总 val：

```text
378/400 = 94.5%
```

总 test：

```text
375/400 = 93.75%
```

---

## 2. 逐级训练结果

### L0

```text
val:  100/100
test: 100/100
```

### L1

```text
val:  75/100
test: 78/100
```

L1 GRPO pilot：

```text
val:  87/100
test: 88/100
```

### L2

```text
val:  70/100
test: 83/100
```

### L3

```text
val:  83/100
test: 78/100
```

### L4

```text
val:  80/100
test: 79/100
```

---

## 3. Checkpoint 保留性

L2 checkpoint：

```text
L0: 100/100, 100/100
L1:  96/100,  94/100
```

L3 checkpoint：

```text
L0: 100/100, 100/100
L1:  97/100,  98/100
L2:  77/100,  88/100
```

L4 checkpoint：

```text
L0: 100/100, 100/100
L1:  97/100,  98/100
L2:  91/100,  92/100
L3:  90/100,  85/100
```

逐级 SFT 没有出现明显的 catastrophic forgetting，最终 L4 模型反而
在 L2/L3 上超过了各自单独 SFT 的 checkpoint。

---

## 4. Baseline

base model 在 L0/L1：

```text
L0: 0/100, 0/100, format errors 100
L1: 0/100, 0/100, format errors 100
```

完整 baseline：

```text
../BASELINE.md
```

---

## 5. 报告索引

SFT：

```text
L0.md
L1.md
L2.md
L3.md
L4.md
```

GRPO：

```text
../GRPO/SCREENING.md
../GRPO/SIGNAL_L1.md
../GRPO/PILOT_L1.md
../GRPO/UNIFIED_CHUNK1.md
```

曲线：

```text
../curves/sft/sft_agent_v3_l0_loss.png
../curves/sft/sft_agent_v3_l1_loss.png
../curves/sft/sft_agent_v3_l2_loss.png
../curves/sft/sft_agent_v3_l3_loss.png
../curves/sft/sft_agent_v3_l4_loss.png
../curves/grpo/grpo_agent_v3_l1_loss.png
```

---

## 6. 运行问题

L2/L3/L4 都出现了训练完成后评估阶段接近 1 小时被后台进程终止：

- checkpoint 均已正常保存；
- 主评估和保留性评估均用 checkpoint 单独补跑；
- GPU 0 上同时存在其他用户训练进程，后续评估已迁移到 GPU 1；
- 没有发现 CUDA OOM 或显存泄漏证据。
