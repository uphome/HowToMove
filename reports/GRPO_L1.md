# L1 GRPO：从 SFT 冷启动继续训练

> 日期：2026-11-17 ｜ 模型：Qwen3-0.6B + LoRA
> 初始化：`/tmp/opencode/sft_l1_600/sft.pt`
> 数据：L1 train maps ｜ 评估：固定 val/test maps ｜ 解码：greedy

---

## 1. 目标

前面已经验证：

- base 直接 L1 GRPO：0 成功，动作几乎只有左右；
- L1 SFT 600：test 72/80 = 90%，四个方向都学会。

本次实验从 L1 SFT checkpoint 初始化，再做 GRPO，验证：

> SFT 先教会模型玩，GRPO 再在非零成功率的基础上继续优化。

---

## 2. 配置

```text
init-adapter: /tmp/opencode/sft_l1_600/sft.pt
levels: L1
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

固定 val/test：

```text
val = 10 张 L1
test = 10 张 L1（训练中）
full test = 80 张 L1（最终）
```

---

## 3. 训练过程

### update 1

```text
success = 8/8
mean_reward = 60.6
collisions = 0
samples = 112
```

### update 2

```text
success = 8/8
mean_reward = 59.95
collisions = 10
samples = 124
```

### update 3

```text
success = 8/8
degenerate = 1
samples = 0
```

### update 4

出现一次明显退化：

```text
success = 0/8
mean_reward = -60.0
collisions = 224
samples = 0
```

### update 5

恢复：

```text
success = 8/8
mean_reward = 60.8
collisions = 0
samples = 0
```

注意：update 3–5 基本都是组内退化，没有产生有效训练样本；真正发生参数更新的是前两次有效组。

---

## 4. 评估结果

### val 10 张

```text
update 0: 10/10
update 5: 10/10
```

SFT 初始化和 GRPO 后都保持 100%。

### test 10 张

| 模型 | success |
|---|---:|
| SFT 600 | 7/10 |
| GRPO 5 updates | **9/10** |

### 完整 test 80 张

| 指标 | SFT 600 | GRPO 5 updates |
|---|---:|---:|
| success | 72/80 = 90.0% | **74/80 = 92.5%** |
| mean steps | 13.75 | **13.30** |
| mean collisions | 2.74 | **1.98** |
| format errors | 0 | 0 |
| optimality | 1.077 | **1.081** |
| final distance | 1.125 | **0.938** |

GRPO 的动作分布：

```text
up=315, down=250, left=301, right=198
```

四个方向都保留，没有退回左右偏置。

---

## 5. 结论

本次实验验证了完整链路：

```text
base LLM
  → SFT：学会四方向和基本多步操作
  → GRPO：利用成功轨迹继续优化
```

具体结果：

- SFT 600 已经解决“会不会走”；
- GRPO 5 updates 在 test 80 上从 90.0% 提升到 92.5%；
- collisions 从 2.74 降到 1.98；
- mean steps 从 13.75 降到 13.30；
- 动作分布保持均衡。

因此：

> **GRPO 不是不适合 L1，而是不适合从零成功率的 base 策略直接开始。**
> SFT 冷启动提供了成功轨迹和四方向覆盖之后，GRPO 才能有效工作。

---

## 6. 局限

- val 只有 10 张，test 80 张；
- 单 seed；
- 5 updates 中有 3 个 update 组内退化，有效更新只有前两次；
- update 4 出现过一次严重退化（0/8，224 collisions）；
- `best` 选择以 val 为准，但 val 在 SFT 和 GRPO 都是 10/10，最终报告使用 `last` GRPO checkpoint；
- 没有 OOD；
- L1 已经较高，提升空间有限。

---

## 7. 下一步

建议：

1. **多 seed 复测** GRPO 的 +2.5pp 是否稳定；
2. 增加 updates / 调整 lr，观察是否还能提升；
3. 对 6 张失败 test 做失败分析；
4. 进入 L2/L3 时继续使用同样的 SFT → GRPO 流程；
5. 评估 OOD/更大迷宫时，单独报告，不混入主指标。

---

## 8. 复现

### 从 SFT checkpoint 训练 GRPO

```bash
python scripts/train_grpo.py \
  --levels L1 \
  --obs-style rel --fewshot rel_l1 \
  --visited --revisit-penalty 0 \
  --updates 5 --maps-per-update 1 --g 8 \
  --lr 5e-5 --temperature 1.2 \
  --max-tokens 48 --max-seq-len 1280 --memory 3 \
  --seed 7 \
  --eval-every 5 --val-limit 10 --test-limit 10 \
  --init-adapter /tmp/opencode/sft_l1_600/sft.pt \
  --json /tmp/opencode/grpo_l1_sft600.json \
  --eval-json /tmp/opencode/grpo_l1_sft600_eval.json \
  --save /tmp/opencode/grpo_l1_sft600/last.pt \
  --best-save /tmp/opencode/grpo_l1_sft600/best.pt
```

### 完整 test 80

```bash
python scripts/train_grpo.py \
  --levels L1 \
  --obs-style rel --fewshot rel_l1 \
  --visited --revisit-penalty 0 \
  --updates 0 --val-limit 0 --test-limit 80 \
  --max-seq-len 1280 --memory 3 \
  --init-adapter /tmp/opencode/grpo_l1_sft600/last.pt \
  --no-ui \
  --json /tmp/opencode/grpo_l1_sft600_test80.json \
  --eval-json /tmp/opencode/grpo_l1_sft600_test80_eval.json
```

产物：

```text
/tmp/opencode/grpo_l1_sft600/last.pt
/tmp/opencode/grpo_l1_sft600.json
/tmp/opencode/grpo_l1_sft600_test80.json
```

相关报告：

```text
reports/SFT_llm.md
reports/L0.md
reports/L1.md
docs/EXPERIMENT_PROTOCOL.md
```
