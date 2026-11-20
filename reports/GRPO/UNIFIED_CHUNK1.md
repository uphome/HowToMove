# Unified GRPO：chunk 1

> 日期：2026-11-20 ｜ 模型：Qwen3-0.6B + LoRA
> 初始化：L4 SFT checkpoint
> 训练数据：`data/grpo/l1_l4_pool.json`（324 hard maps）
> 脚本：`scripts/train_grpo_agent.py`

---

## 0. GRPO pool 筛选摘要

用最终 L4 SFT checkpoint 对 L1–L4 全部 3200 张 train maps 各跑一条
greedy 轨迹：

| level | screened | easy | hard |
|---|---:|---:|---:|
| L1 | 800 | 771 | 29 |
| L2 | 800 | 755 | 45 |
| L3 | 800 | 675 | 125 |
| L4 | 800 | 675 | 125 |
| 合计 | 3200 | 2876 | **324** |

选择规则：

```text
greedy success == false
→ unified GRPO pool
```

详细筛选流程见 [`SCREENING.md`](SCREENING.md)。

---

## 1. 训练结果

本次只训练了 pool 的前 20 个 update：

```text
updates:      20
G:            8
temperature:  0.8
lr:           1e-5
beta KL:      0
elapsed:      约 20 min
```

checkpoint：

```text
/tmp/opencode/grpo_unified_chunk1/grpo_agent.pt
```

## 2. 完整 val/test

| level | L4 SFT val/test | GRPO chunk1 val/test |
|---|---:|---:|
| L0 | 100/100, 100/100 | 100/100, 100/100 |
| L1 | 97/100, 98/100 | **98/100, 97/100** |
| L2 | 91/100, 92/100 | **92/100, 91/100** |
| L3 | 90/100, 85/100 | **90/100, 87/100** |
| L4 | 80/100, 79/100 | 80/100, 79/100 |

变化很小：

```text
L1: val +1, test -1
L2: val +1, test -1
L3: val 0, test +2
L4: 不变
```

说明 20 个 update 还不足以明显改善困难墙体策略。

## 3. 训练曲线

Reward：

![GRPO reward](../curves/grpo/grpo_unified_chunk1_reward.png)

Reward range：

![GRPO reward range](../curves/grpo/grpo_unified_chunk1_reward_range.png)

KL：

![GRPO KL](../curves/grpo/grpo_unified_chunk1_kl.png)

Loss：

![GRPO loss](../curves/grpo/grpo_unified_chunk1_loss.png)

Success：

![GRPO success](../curves/grpo/grpo_unified_chunk1_success.png)

当前 `beta=0`，KL 仅作为诊断曲线，不参与 loss 约束；KL 采样估计可以为负。

## 4. 推理可视化

每个 level 从 test 中选 5 张 map；L1–L4 优先选择失败样本，L0 使用成功样本。
红色叉表示碰撞，蓝色线是实际执行路径，数字是 step。

L0：

![L0 rollouts](../curves/rollouts/grpo_chunk1_rollout_L0.png)

L1：

![L1 rollouts](../curves/rollouts/grpo_chunk1_rollout_L1.png)

L2：

![L2 rollouts](../curves/rollouts/grpo_chunk1_rollout_L2.png)

L3：

![L3 rollouts](../curves/rollouts/grpo_chunk1_rollout_L3.png)

L4：

![L4 rollouts](../curves/rollouts/grpo_chunk1_rollout_L4.png)

GIF：

```text
../curves/rollouts/grpo_chunk1_rollout_L0.gif
../curves/rollouts/grpo_chunk1_rollout_L1.gif
../curves/rollouts/grpo_chunk1_rollout_L2.gif
../curves/rollouts/grpo_chunk1_rollout_L3.gif
../curves/rollouts/grpo_chunk1_rollout_L4.gif
```

### 典型失败轨迹

| level | map | steps | collisions | actions |
|---|---|---:|---:|---|
| L1 | L1-361d91521e | 30 | 23 | down=28, left=1, right=1 |
| L1 | L1-af9327c158 | 30 | 24 | left=28, up=2 |
| L2 | L2-193fc1e11c | 30 | 28 | up=27, left=3 |
| L2 | L2-3bc54dd42c | 30 | 25 | left=16, down=13 |
| L3 | L3-3b1f6faacc | 30 | 29 | up=22, right=8 |
| L3 | L3-5b0d673443 | 30 | 28 | down=18, right=11 |
| L4 | L4-23362226b3 | 30 | 29 | up=29, left=1 |
| L4 | L4-2c5c58f39c | 30 | 28 | left=27, down=3 |

## 5. 可视化观察

L1–L4 的失败样本都表现为：

```text
30 steps
大量 collision
路径在局部区域内反复来回
```

例如：

```text
L2 失败样本: 30 steps, 16–28 collisions
L3 失败样本: 30 steps, 17–29 collisions
L4 失败样本: 30 steps, 23–29 collisions
```

L0 的 5 个样本全部成功，且碰撞很少。

## 6. 限制

- 只训练了 20 / 324 hard maps；
- 只做了 20 个 update；
- 没有做 KL 约束；
- 训练和最终评估分在不同进程中完成；
- 当前结果是 GRPO chunk1 的阶段性结果，不是最终统一 GRPO。

## 7. 下一步

继续按 pool offset 训练下一批 chunk：

```text
chunk2: offset=20
chunk3: offset=40
...
```

每个 chunk 保存 checkpoint 和 metrics，最后再统一评估。

复现 chunk1：

```bash
python scripts/train_grpo_agent.py \
  --init-adapter /tmp/opencode/sft_agent_v3_l4/sft_agent.pt \
  --pool data/grpo/l1_l4_pool.json \
  --offset 0 \
  --updates 20 \
  --g 8 \
  --temperature 0.8 \
  --top-p 0.95 \
  --lr 1e-5 \
  --no-eval \
  --save-dir /tmp/opencode/grpo_unified_chunk1 \
  --json /tmp/opencode/grpo_unified_chunk1.json
```
