# 实验协议（后续阶段沿用）

> 适用于 L0/L1/L2… 的单步与多步实验。
> 目标：不同 run、不同观测编码、不同阶段之间可比较。

---

## 1. 数据口径

### 单步

- 训练：`data/states/train.jsonl`
- 早停 / best：`data/states/val.jsonl`
- 最终报告：`data/states/test.jsonl`
- 真实分布对照：`data/eval_states.jsonl`（natural）

状态集必须按动作类别均衡：

- `singleton_up/down/left/right`
- `mixed_dr/dl/ur/ul`

新增/修改状态集后必须运行：

```bash
python scripts/check_states.py
```

不通过不训练。

### 多步

- 地图 split 不变；
- 评估集固定，不从训练图临时生成；
- 后续如加入 start/trajectory 状态集，必须单独命名，不与 single-step 混用。

---

## 2. 评测口径

- 解码：**greedy**（`temperature=0`）；
- 同一组固定 eval 状态；
- 不在 test 上早停、不按 test 选 best；
- val 用于早停/选模；
- test 只在最终报告时使用；
- natural 只作为辅助对照，不用于选模；
- 记录 eval 状态数；
- 单 seed 只能作为初步结果；需要正式结论时至少 2–3 个 seed。

---

## 3. 指标口径

### 主指标

```text
balanced test optimal_rate
```

### 辅助指标

```text
natural test optimal_rate
```

### 方向指标

```text
singleton_accuracy.up/down/left/right
mixed_optimal_rate
```

### 护栏

```text
valid_rate >= 0.99
degenerate_rate
farther_rate / blocked_rate / invalid_rate
ratio_min/max, clip_frac, drop_frac
```

### 训练指标

loss / pg / kl 只用于诊断，不用于判断模型变好。

---

## 4. 对照实验

观测编码 A/B/C 必须：

- 同一 base model；
- 同一 state 集；
- 同一训练配置（updates、states、G、lr、T、seed）；
- 同一 eval 集；
- 都不早停，或都使用相同早停规则；
- 记录 prompt token 数；
- 至少报告 baseline / 中期 / 最终三个点。

当前观测：

| 组 | 说明 |
|---|---|
| absolute | `--obs-style minimal` |
| relative | `--obs-style rel` |
| relative + few-shot | `--obs-style rel --fewshot rel` |

---

## 5. 产物

每次实验保存：

```text
save-dir/
├── ckpt_uXXXX.pt
├── best.pt
└── last.pt
<name>.jsonl
<name>.json
<name>.log
```

报告中必须写：

- 训练配置；
- 数据规模；
- baseline / final；
- 主指标、方向指标、护栏；
- 已知局限；
- 是否达到当前阶段完成标准。

---

## 6. 阶段推进标准

### L0

- balanced test optimal ≥ 90%
- natural test optimal ≥ 90%
- 四个 singleton 方向 ≥ 90%
- valid ≥ 99%
- degenerate 高 → L0 饱和，停止继续堆 update

### L1（带墙单步）

- 同上，但数据换成 L1 balanced 状态集；
- 额外观察 `walls_rel` / 墙坐标是否被理解；
- natural 不能明显低于 absolute baseline。

### 多步

- success rate；
- optimality = steps / shortest（成功局）；
- collisions / format errors；
- 不能只看 mean reward。

---

## 7. 报告模板

每个阶段一份 `reports/<stage>.md`：

1. 阶段目标
2. 数据与评测口径
3. 训练配置
4. 观测对照
5. 结果
6. 是否达标
7. 关键发现
8. 局限
9. 下一阶段
10. 复现命令
