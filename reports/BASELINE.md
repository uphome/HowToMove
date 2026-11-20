# Agent Base Model Baseline

> 日期：2026-11-18 ｜ 模型：Qwen3-0.6B
> Adapter：无 ｜ 解码：greedy ｜ 数据：map-only v3
> Protocol：静态地图 + `move` tool + history_window=12

---

## 1. 结论

base model 在 map-only v3 的 L0/L1 上都完全没有完成任务：

```text
val:  0/200
test: 0/200
```

并且每个 episode 都出现一次格式错误：

```text
format_errors = 100 / 100 tasks per level
```

这不是“路径规划不够好”，而是模型还没有稳定学会使用工具调用格式和持续执行
`move`，因此 SFT 仍然是必要的冷启动步骤。

---

## 2. 配置

```text
base:          Qwen3-0.6B
adapter:       none
history:       12
max tokens:    48
temperature:   0
levels:        L0, L1
val maps:      100 per level
test maps:     100 per level
```

评估使用与 SFT 完全相同的消息协议：

```text
system: 静态地图
user:   Begin the maze.
assistant: move(direction)
tool: compact JSON
```

---

## 3. L0

| split | n | success | success rate | mean steps | collisions | format errors |
|---|---:|---:|---:|---:|---:|---:|
| val | 100 | 0 | 0.0% | 10.6 | 22 | 100 |
| test | 100 | 0 | 0.0% | 10.7 | 20 | 100 |

val 动作分布：

```text
up=253, down=243, left=241, right=220
```

test 动作分布：

```text
up=253, down=246, left=248, right=220
```

---

## 4. L1

| split | n | success | success rate | mean steps | collisions | format errors |
|---|---:|---:|---:|---:|---:|---:|
| val | 100 | 0 | 0.0% | 10.1 | 58 | 100 |
| test | 100 | 0 | 0.0% | 10.1 | 52 | 100 |

val 动作分布：

```text
up=244, down=239, left=228, right=199
```

test 动作分布：

```text
up=242, down=238, left=225, right=208
```

---

## 5. 汇总

| split | n | success | mean steps | collisions | format errors |
|---|---:|---:|---:|---:|---:|
| val | 200 | 0 | 10.3 | 80 | 200 |
| test | 200 | 0 | 10.4 | 72 | 200 |

完整结果：

```text
/tmp/opencode/baseline_agent_l0_l1.json
```

---

## 6. 复现

```bash
python scripts/sft_agent.py \
  --eval-only \
  --levels L0,L1 \
  --data data \
  --val-limit 100 \
  --test-limit 100 \
  --eval-batch-size 8 \
  --json /tmp/opencode/baseline_agent_l0_l1.json
```
