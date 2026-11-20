# Unified GRPO Map Screening（v3）

> 日期：2026-11-19 ｜ policy：L4 SFT checkpoint
> 数据：map-only v3，train split
> 脚本：`scripts/screen_grpo_maps.py`

---

## 1. 筛选结果

用最终 L4 SFT checkpoint 对 L1–L4 全部 train maps 各跑一条 greedy 轨迹：

```text
screened maps: 3200
easy:          2876
hard:           324
format errors:    2
```

按 level：

| level | n | easy | hard | format errors | mean collisions | mean steps |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 800 | 771 | 29 | 0 | 0.88 | 12.46 |
| L2 | 800 | 755 | 45 | 0 | 1.43 | 14.24 |
| L3 | 800 | 675 | 125 | 0 | 4.22 | 16.17 |
| L4 | 800 | 675 | 125 | 2 | 3.95 | 16.39 |
| 合计 | 3200 | 2876 | 324 | 2 | — | — |

## 2. GRPO pool

筛选规则：

```text
greedy success == false
→ 进入 unified GRPO pool
```

不再做单独 G=8 预探测；GRPO 训练时直接对 pool 内 map 采样 G=8，
遇到组内 reward 没有差异的 group 直接 skip。

pool：

```text
data/grpo/l1_l4_pool.json
```

分布：

```text
L1: 29
L2: 45
L3: 125
L4: 125
total: 324
```

## 3. 为什么可以直接 GRPO

- greedy 已经执行了一次真实 Agent rollout；
- 失败 map 至少说明当前策略没有一次成功；
- GRPO 内部的 G=8 会再次产生多条轨迹；
- 如果某个 group 全部失败且 reward 完全一样，则 advantage≈0，跳过；
- 如果 group 有不同碰撞/距离/reward，则仍可学习。

## 4. 估计训练量

当前 L1 GRPO pilot：

```text
10 updates ≈ 19.5 min
≈ 2 min/update
```

如果 pool 内每张 map 做一次 update：

```text
324 maps × ≈2 min ≈ 10–11 h
```

因此实际训练需要拆成多个 chunk：

```text
每 chunk 30–50 updates
保存 checkpoint
单独评估
```

避免再次触发长时间后台任务上限。

## 5. 复现

```bash
python scripts/screen_grpo_maps.py \
  --init-adapter /tmp/opencode/sft_agent_v3_l4/sft_agent.pt \
  --levels L1 \
  --split train \
  --batch-size 8 \
  --resume \
  --rows-jsonl /tmp/opencode/grpo_screen_L1.rows.jsonl \
  --out /tmp/opencode/grpo_screen_L1.json
```

其余 level 同理。

合并 pool：

```bash
python scripts/build_grpo_pool.py \
  --rows /tmp/opencode/grpo_screen_L1.rows.jsonl \
         /tmp/opencode/grpo_screen_L2.rows.jsonl \
         /tmp/opencode/grpo_screen_L3.rows.jsonl \
         /tmp/opencode/grpo_screen_L4.rows.jsonl \
  --out data/grpo/l1_l4_pool.json
```
