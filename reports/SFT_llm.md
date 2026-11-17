# SFT 冷启动：让 LLM 先学会玩多步迷宫

> 日期：2026-11-17 ｜ 模型：Qwen3-0.6B + LoRA
> 训练脚本：`scripts/sft_multistep.py`
> 数据：L0 / L1 地图的 BFS 最短路专家轨迹

---

## 核心结论

一开始，base LLM 实际上还不会玩这个多步迷宫游戏。

在 L1 多步 rollout 中，它几乎只输出 `left/right`。即使 goal 在正上方，
也会持续左右移动或撞墙。此时直接接入 GRPO，一轮 8 条采样轨迹里
既没有成功轨迹，也没有 `up/down` 动作覆盖。组内奖励没有有效差异，
advantage 接近噪声，策略会进一步塌缩成单一方向。

因此，**从零开始直接多步 GRPO 效果不好，并不是 GRPO 算法本身无效，
而是当前策略没有提供可学习的成功样本和动作覆盖。**

SFT 的作用是先教会模型这个游戏的基本操作：

- 四个方向都能输出；
- 能读取 goal delta、墙和边界；
- 能沿 BFS 最短路走多步；
- 初步学会遇到墙后绕路。

SFT 之后，模型在 L0/L1 上获得了非零成功率。此时 8 条 rollout 中
才开始出现成功轨迹，组内 reward 才有区分度，GRPO 才具备正确的
优化起点。

整体流程应当是：

```text
base LLM
  → SFT 冷启动：学会四方向和基本多步操作
  → GRPO 微调：利用成功轨迹进一步优化策略
```

需要强调：

```text
SFT 后“具备 GRPO 条件”
≠
SFT 后“GRPO 一定会提升”
```

目前已经验证的是：

- base 直接 L1 GRPO 失败；
- SFT 能把 L1 拉到 90% test 成功率；
- 从 SFT checkpoint 接 GRPO 仍然是下一步实验。

---

## 1. 为什么需要 SFT

### 1.1 base 模型的多步行为

在 L1 多步 rollout 中观察到：

```text
success = 0/10
动作大多只有 left/right
大量撞墙或左右震荡
```

典型轨迹：

```text
left collision
left collision
right move
left move
left collision
...
```

模型不是不会输出工具调用格式，而是**不会把当前状态稳定映射到正确方向**。

### 1.2 直接 GRPO 为什么没有信号

GRPO 依赖同一张地图内 8 条轨迹的奖励差异：

```text
8 条轨迹
→ 总奖励
→ 组内标准化 advantage
→ 提高好轨迹的概率
```

如果 8 条轨迹都失败，或者都只输出左右：

- 没有成功轨迹可以强化；
- 没有 `up/down` 动作样本；
- advantage 很小或完全退化；
- 策略会继续偏向左右，甚至塌缩成单一动作。

这不是 GRPO 公式的问题，而是**起点策略没有探索到正确动作**。

### 1.3 SFT 提供了什么

SFT 直接给每个状态提供 BFS 专家动作：

```text
状态 s_t
目标动作 a_t* = 最短路上的下一步
loss = -log π(a_t* | s_t)
```

它不需要模型先碰巧成功，也不需要等待成功奖励出现。

---

## 2. SFT 数据生成

训练脚本：

```text
scripts/sft_multistep.py
```

### 2.1 生成流程

```text
地图
→ BFS 最短路 shortest_path()
→ 路径上的每个状态
→ render_observation(rel + last + visited_rel)
→ 目标动作
→ tool_call completion
```

对路径上的每个相邻位置：

```python
action = direction_between(pos, next_pos)
prompt = render_observation(
    maze, pos, last_actions,
    style="rel",
    visited=path[:i],
)
target = format_tool_call(action) + eos
```

### 2.2 观测格式

SFT 使用和 rollout 完全相同的观测函数，避免训练/推理格式不一致。

L0 使用：

```text
--obs-style rel
--fewshot rel
```

L1 使用：

```text
--obs-style rel
--fewshot rel_l1
```

prompt 中包含：

- `goal: delta=(dx,dy)`；
- `bounds`；
- `walls_rel`；
- `last`；
- `visited_rel`。

### 2.3 数据规模

#### L0 SFT

```text
maps = 100
samples = 1110
epochs = 1
actions:
  up    = 331
  down  = 245
  left  = 282
  right = 252
```

prompt 长度范围：

```text
852 .. 918
```

#### L1 SFT

100 张图版本：

```text
samples = 1119
prompt 长度范围 = 1070 .. 1181
```

600 张图版本：

```text
maps = 600
samples = 6674
epochs = 2
actions:
  up    = 1688
  down  = 1552
  left  = 1736
  right = 1698
```

prompt 长度范围：

```text
1070 .. 1183
```

---

## 3. 训练配置

```text
base: Qwen3-0.6B
adapter: LoRA r=16
loss: token-level cross entropy
batch size: 8
lr: 5e-5
grad clip: 1.0
```

### 3.1 按 prompt 长度分桶

不同状态的 `visited_rel` 长度不同，prompt 长度会变化。

训练时按 prompt 长度分桶：

```text
同一 bucket 内 prompt 长度完全一致
→ 一个 batch 内不需要处理变长 prompt 的 left/right padding
→ 只对 completion token 计算 loss
```

### 3.2 LoRA 初始化

L1 600 使用 L0 SFT checkpoint 初始化：

```text
--init-adapter /tmp/opencode/sft_l0_100/sft.pt
```

这样模型先具备四方向和多步基础，再学习 L1 的墙和绕路。

### 3.3 评估

- greedy；
- 固定 val/test 地图；
- 记录 success、steps、collisions、format errors、动作分布；
- 训练与评估使用同一个 `render_observation` / `render_prompt`。

---

## 4. 结果

### 4.1 L0 SFT

```text
maps = 100
samples = 1110
epochs = 1

val:  10/10
test: 10/10
collisions: 0
```

test 动作分布：

```text
up=31, down=22, left=27, right=23
```

L0 已经能稳定走完，并且四个方向都出现。

### 4.2 L1 SFT：100 张图不够

从 base 直接 SFT：

```text
val:  0/10
test: 1/10
collisions: 254 / 277
```

从 L0 SFT checkpoint 继续：

```text
val:  0/10
test: 0/10
```

典型失败：

```text
一个地图固定一个方向 30 步
→ 撞墙
→ 不会根据墙转弯
```

100 张 L1 图不足以覆盖墙、绕路和不同地图的组合。

### 4.3 L1 SFT：600 张图 + 2 epochs

配置：

```text
maps = 600
samples = 6674
epochs = 2
init = L0 SFT
```

结果：

```text
val:  10/10
test: 72/80 = 90.0%
```

完整 test 80 张的指标：

```text
mean steps: 13.75
mean collisions: 2.74/局
format errors: 0
actions:
  up    = 282
  down  = 272
  left  = 301
  right = 245
optimality: 1.077
final distance: 1.125
```

success test 10 张的小样本结果是 7/10；扩到完整 80 张后是 72/80，
说明 600 张版本确实学到了可泛化的 L1 多步策略。

### 4.4 对比

| 实验 | 训练规模 | val | test |
|---|---:|---:|---:|
| L1 GRPO from base | 5 updates | 0/10 | 0/10 |
| L1 SFT | 100 图 / 1 epoch | 0/10 | 1/10 |
| L1 SFT + L0 init | 100 图 / 1 epoch | 0/10 | 0/10 |
| L1 SFT + L0 init | 600 图 / 2 epochs | **10/10** | **72/80** |

---

## 5. 关键发现

1. **SFT 能解决探索问题**
   - 每个状态都有专家动作；
   - 不依赖偶然成功；
   - 直接覆盖 up/down/left/right。

2. **L0 和 L1 的数据需求不同**
   - L0 100 张图已经足够；
   - L1 100 张图明显过拟合/覆盖不足；
   - L1 600 张图 + 2 epochs 才达到 90%。

3. **课程初始化有效**
   - 先 L0 SFT，再 L1 SFT；
   - 比从 base 直接学 L1 更稳定。

4. **动作均衡不是唯一问题**
   - SFT 数据动作本身已经均衡；
   - 100 张 L1 仍然会固定方向；
   - 关键是墙、绕路和状态覆盖。

5. **SFT 不是 GRPO 的替代品**
   - SFT 负责“会走”；
   - GRPO 负责在已有成功率上继续优化；
   - 从零直接 GRPO 不适合当前多步任务。

---

## 6. 局限

- L1 val 只有 10 张；test 已扩到 80 张；
- 单 seed；
- 没有 OOD（8×8 / 16×16）；
- 只使用 BFS 最短路轨迹，没有使用失败状态和 DAgger；
- L1 600 / 2 epochs 训练约 38 分钟；
- 还有 8 张 test 失败，表现为固定方向撞墙或局部循环；
- 从 SFT checkpoint 接 GRPO 的最终提升尚未验证。

---

## 7. 下一步：从 SFT 接 GRPO

现在 L1 SFT 已经有 90% test 成功率，具备 GRPO 微调条件。

建议第一轮：

```text
init-adapter = /tmp/opencode/sft_l1_600/sft.pt
levels = L1
obs-style = rel
fewshot = rel_l1
visited = true
updates = 5
maps-per-update = 1
G = 8
lr = 5e-5
max_seq_len = 1280
```

重点对照：

```text
success
collisions
mean steps
optimality
action distribution
```

如果 GRPO 不能提升，再考虑：

- 用 SFT 的 8 个失败案例做额外专家轨迹；
- DAgger：在模型实际走到的错误状态上补专家动作；
- 课程学习：L0 → L1 → L2；
- 对失败状态做困难样本重采样。

---

## 8. 复现命令

### L0 SFT

```bash
python scripts/sft_multistep.py \
  --levels L0 --maps 100 --epochs 1 \
  --obs-style rel --fewshot rel --visited \
  --max-seq-len 1024 \
  --save-dir /tmp/opencode/sft_l0_100
```

### L1 SFT（100 张，失败对照）

```bash
python scripts/sft_multistep.py \
  --levels L1 --maps 100 --epochs 1 \
  --obs-style rel --fewshot rel_l1 --visited \
  --max-seq-len 1280 \
  --save-dir /tmp/opencode/sft_l1_100
```

### L1 SFT（600 张，最终版本）

```bash
python scripts/sft_multistep.py \
  --levels L1 --maps 600 --epochs 2 \
  --obs-style rel --fewshot rel_l1 --visited \
  --max-seq-len 1280 \
  --init-adapter /tmp/opencode/sft_l0_100/sft.pt \
  --save-dir /tmp/opencode/sft_l1_600
```

### 用 SFT checkpoint 跑完整 test 80

```bash
python scripts/train_grpo.py \
  --levels L1 --obs-style rel --fewshot rel_l1 \
  --updates 0 --val-limit 0 --test-limit 80 \
  --max-seq-len 1280 --memory 3 --visited \
  --init-adapter /tmp/opencode/sft_l1_600/sft.pt \
  --no-ui \
  --json /tmp/opencode/sft_l1_600_test80.json \
  --eval-json /tmp/opencode/sft_l1_600_test80_eval.json \
  --save /tmp/opencode/sft_l1_600_test80/last.pt \
  --best-save /tmp/opencode/sft_l1_600_test80/best.pt
```

---

## 9. 产物

```text
/tmp/opencode/sft_l0_100/sft.pt
/tmp/opencode/sft_l1_600/sft.pt
/tmp/opencode/sft_l1_600_test80.json
/tmp/opencode/sft_l1_600_test80_eval.json
```

代码：

```text
scripts/sft_multistep.py
scripts/train_grpo.py
grpo/multistep.py
env/render.py
```

相关报告：

```text
reports/L0.md
reports/L1.md
reports/SFT_llm.md
docs/EXPERIMENT_PROTOCOL.md
```
