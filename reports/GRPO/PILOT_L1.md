# Agent GRPO L1：从 SFT checkpoint 继续训练

> 日期：2026-11-18 ｜ 模型：Qwen3-0.6B + LoRA
> 数据：map-only v3，L1 train
> 初始化：L1 SFT e2 checkpoint

---

## 1. 结论

GRPO 在 L1 SFT checkpoint 上带来了明显提升：

| 模型 | val | test | mean steps | collisions | format errors |
|---|---:|---:|---:|---:|---:|
| base | 0/100 | 0/100 | 10.1 / 10.1 | 58 / 52 | 100 / 100 |
| L1 SFT | 75/100 | 78/100 | 17.4 / 17.0 | 765 / 685 | 0 / 0 |
| **L1 SFT + GRPO** | **87/100** | **88/100** | **15.4 / 15.4** | **461 / 443** | **0 / 0** |

GRPO 同时改善了：

- success rate；
- 平均步数；
- 碰撞次数；
- 失败轨迹中的超时/绕路行为。

---

## 2. 训练设置

```text
checkpoint:   /tmp/opencode/sft_agent_v3_l1_e2/sft_agent.pt
level:        L1
maps:         前 10 张 train map
G:            8
temperature:  0.8
top_p:        0.95
lr:           1e-5
clip_eps:     0.2
beta KL:      0.0
max grad:     1.0
history:      12
max seq len:  4096
updates:      10
elapsed:      19.5 min
```

每张训练 map 采样 8 条完整 Agent 轨迹，按组内总 reward 做标准化，
再把每条轨迹的每个 turn 转成 GRPO token sample。

---

## 3. 组内信号

训练前对 20 张 L1 train map 做 G=8 探测：

```text
mixed groups:    5/20  (25%)
all-fail groups: 0/20
all-success:     15/20
mean reward range: 31.14
```

mixed 组示例：

| map | success | reward range |
|---|---:|---:|
| L1-0148318833 | 6/8 | 116.0 |
| L1-01ed9941af | 7/8 | 84.7 |
| L1-03d07e8620 | 7/8 | 113.6 |
| L1-059e65c513 | 6/8 | 121.4 |
| L1-0629721cfb | 6/8 | 69.8 |

因此 GRPO 有可用的 group-relative signal。

---

## 4. 更新统计

10 个 update 全部产生非退化组：

```text
update 1:  success 8/8, samples 97, loss 0.0714
update 2:  success 8/8, samples 107, loss 0.1277
update 3:  success 6/8, samples 124, loss 0.5399
update 4:  success 8/8, samples 75, loss 0.0516
update 5:  success 8/8, samples 167, loss 0.1758
update 6:  success 8/8, samples 85, loss 0.1081
update 7:  success 7/8, samples 100, loss 0.5290
update 8:  success 8/8, samples 92, loss 0.3770
update 9:  success 8/8, samples 177, loss 0.0275
update 10: success 8/8, samples 121, loss 0.0930
```

注意：成功组也可能因为路径长度、碰撞次数和 step reward 不同而有
advantage；因此并不要求每个组都包含成功和失败。

![GRPO update loss](curves/grpo_agent_v3_l1_loss.png)

---

## 5. 评估

使用与 SFT 相同协议：

```text
static map system
Begin the maze.
move tool
compact JSON
```

greedy 解码，`history_window=12`，batch=8。

动作分布：

```text
val:  up=320, down=431, left=333, right=460
test: up=366, down=365, left=302, right=507
```

GRPO 后仍有失败：

```text
val failures:  13
test failures: 12
```

主要是少数 hard map 上的持续撞墙/30 步超时，但数量已明显减少。

---

## 6. 限制

- 这次只用了前 10 张 L1 train map 做 10 个 update；
- 还不是全量训练；
- 训练和评估都仍是 12×12；
- GRPO 的收益可能部分来自对训练 map 的重复优化，需要在更多 map 上验证；
- 尚未测试 GRPO 从 L0 或更大训练集开始的版本。

---

## 7. 复现

```bash
python scripts/train_grpo_agent.py \
  --init-adapter /tmp/opencode/sft_agent_v3_l1_e2/sft_agent.pt \
  --level L1 \
  --updates 10 \
  --g 8 \
  --temperature 0.8 \
  --top-p 0.95 \
  --lr 1e-5 \
  --val-limit 100 \
  --test-limit 100 \
  --save-dir /tmp/opencode/grpo_agent_v3_l1 \
  --json /tmp/opencode/grpo_agent_v3_l1.json
```

完整结果：

```text
/tmp/opencode/grpo_agent_v3_l1.json
```
