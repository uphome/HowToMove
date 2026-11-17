# L2 GRPO：从 L1 checkpoint 直接微调

> 日期：2026-11-17 ｜ 模型：Qwen3-0.6B + LoRA
> 初始化：`/tmp/opencode/grpo_l1_sft600/last.pt`
> 数据：L2 train maps ｜ 最终评估：固定 L2 test_id 80 张
> 解码：greedy ｜ 评估：eval batch 8

---

## 1. 目标

L1 已经完成：

```text
L1 SFT 600: test 72/80 = 90.0%
L1 SFT → GRPO: test 74/80 = 92.5%
```

L2 的问题是：

> L1 策略能否直接迁移到 L2，并在 L2 上直接 GRPO，而不做 L2 SFT？

先做 L2 zero-shot：

```text
L2 val 10: 5/10
```

说明起点不是 0，可以直接尝试 GRPO。

---

## 2. 配置

```text
init-adapter: /tmp/opencode/grpo_l1_sft600/last.pt
levels: L2
obs-style: rel
fewshot: rel_l1
visited: true
memory: 3
updates: 5
maps-per-update: 1
G: 8
lr: 5e-5
temperature: 1.2
max_tokens: 48
max_seq_len: 1280
seed: 7
revisit-penalty: 0
```

评估：

```text
val: 10 张 L2
test: 10 张快速评估
最终: 80 张完整 L2 test_id
```

---

## 3. 训练过程

```text
update 1: success 8/8, collisions 0
update 2: success 0/8, collisions 208, degenerate
update 3: success 0/8, collisions 224, degenerate
update 4: success 5/8, collisions 57, samples 126
update 5: success 1/8, collisions 191, samples 229
```

训练仍然不稳定，update 2/3 发生退化；最后 val 从 5/10 提升到 6/10。

---

## 4. 完整 test 80 结果

| 模型 | success | mean steps | collisions | optimality |
|---|---:|---:|---:|---:|
| L2 zero-shot | 32/80 = 40.0% | 23.13 | 15.96 | 1.069 |
| L2 GRPO 5 updates | **33/80 = 41.2%** | **22.83** | **15.59** | 1.070 |

动作分布：

```text
zero-shot:
  up=241, down=531, left=846, right=232

GRPO:
  up=214, down=452, left=945, right=215
```

结论：

- GRPO 只多成功 1/80；
- collisions 和 steps 有很小改善；
- 动作分布仍然偏左/下；
- 这不能算实质提升。

之前 test10 的 `5/10 vs 4/10` 是噪声；完整 test80 更可靠。

---

## 5. Eval batch

这次新增了 `--eval-batch`：

```text
--eval-batch 8
```

实现：

- `run_maps_batch()` 同时跑多张地图的 1 条 greedy 轨迹；
- 每一步把所有未结束地图的当前观测拼成一个 batch；
- 地图之间可以有不同的终止步数；
- 不改变 greedy 指标口径。

性能：

```text
L2 test80 serial: 约 15 min
L2 test80 eval-batch=8: 约 5.1–5.3 min
```

约 **2.8×** 加速。

注意：eval batch 只改变评估吞吐，不改变模型或指标。

---

## 6. 结论

```text
L2 可以直接从 L1 checkpoint 接 GRPO
≠
L2 GRPO 一定有效
```

本次结果：

- 起点不是 0，直接 GRPO 技术上可行；
- 5 updates 没有带来稳定提升；
- 训练过程仍有退化和动作偏置；
- 不能据此说“L2 不需要 SFT”。

更准确的结论：

> L2 具备直接 GRPO 的起点，但当前 5-update GRPO 不足以解决问题；需要多 seed、L2 SFT/DAgger 或更稳定的训练策略。

---

## 7. 下一步

1. 多 seed 复测 L2 zero-shot vs GRPO；
2. 对 47 张失败 test 做失败分析；
3. 用 L2 SFT / DAgger 补墙和绕路；
4. 降低 GRPO lr 或减少每 update 的更新量，避免 update 2/3 退化；
5. 继续使用 `--eval-batch 8`，所有后续实验统一用完整 test80。

---

## 8. 复现

### L2 zero-shot

```bash
python scripts/train_grpo.py \
  --levels L2 --obs-style rel --fewshot rel_l1 \
  --visited --revisit-penalty 0 \
  --updates 0 --val-limit 10 --test-limit 10 \
  --max-seq-len 1280 --memory 3 \
  --init-adapter /tmp/opencode/grpo_l1_sft600/last.pt \
  --no-ui
```

### L2 GRPO 5 updates

```bash
python scripts/train_grpo.py \
  --levels L2 --obs-style rel --fewshot rel_l1 \
  --visited --revisit-penalty 0 \
  --updates 5 --maps-per-update 1 --g 8 \
  --lr 5e-5 --temperature 1.2 \
  --max-tokens 48 --max-seq-len 1280 --memory 3 \
  --eval-every 5 --val-limit 10 --test-limit 10 \
  --init-adapter /tmp/opencode/grpo_l1_sft600/last.pt
```

### 完整 test80 + eval batch

```bash
python scripts/train_grpo.py \
  --levels L2 --obs-style rel --fewshot rel_l1 \
  --visited --revisit-penalty 0 \
  --updates 0 --val-limit 0 --test-limit 80 \
  --eval-batch 8 \
  --max-seq-len 1280 --memory 3 \
  --init-adapter /tmp/opencode/grpo_l2_from_l1/last.pt \
  --no-ui
```

产物：

```text
/tmp/opencode/l2_zero_test80_batch.json
/tmp/opencode/l2_grpo_test80_batch.json
/tmp/opencode/l2_full_test80_batch/
```
