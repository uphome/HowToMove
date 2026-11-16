# 相对观测 A/B 报告

> 日期：2026-11-16 ｜ 目的：验证「去掉绝对坐标、改成全局相对坐标」是否可行
> 数据：balanced L0（train 320 / val 64 / test 64） + natural L0 60
> 训练：10 updates，states=16，G=8，lr=5e-5，T=1.2，seed=7
> 评估：greedy

---

## 1. 结论

**当前这套 signed-delta 全局相对格式不可用。**

- 基座模型对 `goal: delta=(...)`、`bounds`、`walls_rel` 基本不接地；
- 两个 eval 集上模型都退化成常量 `right`；
- 训练组从第 7 个 update 起几乎全部退化，GRPO 没有梯度信号；
- 相对组 10 个 update 后 balanced test optimal 仍是 **31.25%**，与 baseline 完全一样。

绝对坐标组在同样配置下 balanced test 从 65.62% 升到 89.06%。  
所以本轮不是“相对坐标一定差”，而是“这个相对编码当前模型读不懂”。

---

## 2. 格式

absolute（`--obs-style minimal`）：

```text
pos=(7,1) goal=(0,3)
walls: none
```

relative（`--obs-style rel`）：

```text
map 12x12, x=right y=down, origin=you
goal: delta=(-7,+2)
bounds: left=7 right=4 up=1 down=10
walls_rel: none
```

prompt 长度（含 chat template + tools）：

| level | minimal | rel | 增量 |
|---|---:|---:|---:|
| L0 | 194.7 | 226.4 | +31.7 |
| L1 | 223.5 | 254.7 | +31.2 |
| L2 | 228.5 | 259.4 | +30.9 |
| L3 | 247.0 | 277.5 | +30.5 |
| L4 | 271.4 | 301.1 | +29.7 |

token 增量约 12–16%，不是主要问题。

---

## 3. 结果

### absolute

| 指标 | baseline | update 10 |
|---|---:|---:|
| balanced val optimal | 64.06% | 95.31% |
| balanced test optimal | 65.62% | **89.06%** |
| natural test optimal | 81.67% | 96.67% |
| test action counts | right/left/up | right 20 / left 20 / down 15 / up 9 |
| test singleton acc up/down/left/right | 100/0/66.7/91.7 | **75/100/91.7/83.3** |

### relative

| 指标 | baseline | update 10 |
|---|---:|---:|
| balanced val optimal | 31.25% | 31.25% |
| balanced test optimal | 31.25% | **31.25%** |
| natural test optimal | 55.00% | 55.00% |
| test action counts | right 64 | right 64 |
| test singleton acc up/down/left/right | 0/0/0/100 | **0/0/0/100** |
| degenerate rate | — | 100% |

相对组从第 7 个 update 开始：

```
states=16 n=0 skip=16
```

模型对所有状态都输出 `right`，组内没有动作差异，advantage 全 0。

---

## 4. 解释

1. **基座模型不熟悉这种抽象格式**
   - `delta=(-7,+2)` 需要模型自己完成“负号 = 左/上”的映射；
   - 绝对格式 `pos=(7,1) goal=(0,3)` 更接近预训练里见过的坐标比较；
   - 结果就是模型直接忽略相对信息，退化为常量 `right`。

2. **GRPO 无法修复“从不采样其他动作”**
   - 如果 G 条采样全是 `right`，组内奖励完全相同；
   - `group_advantages` 返回全 0；
   - 没有梯度，训练不会改变策略。

3. **相对坐标不是错误方向，而是编码不合适**
   - 7×7 局部视野、方向词、自然语言描述可能更容易；
   - 或者需要 SFT 冷启动，先教会模型读 relative 格式。

额外 zero-shot 探针：把 `delta=(-7,+2)` 换成方向词

```text
exit: 7 left and 2 down
```

基座模型仍然对 64/64 个 balanced test 状态全部输出 `right`
（natural 60/60 也全是 right）。说明问题不只是“负号/数字”，
而是模型完全没有把相对描述接到动作空间上；当前阶段必须 SFT 或换空间编码。

---

## 5. 建议下一步

按优先级：

1. **方向词相对格式**
   ```text
   exit: left 7, down 2
   ```
   或
   ```text
   exit is 7 cells to the left and 2 cells down
   ```
2. **局部 7×7 视野 + goal delta**
   - 用 `A/#/./~` 表示 agent、墙、自由、边界；
   - 目标在视野外时给 delta。
3. **保留绝对 `pos`、只把 goal/墙改成相对**
   - 用于隔离“没有绝对锚点”是不是主因。
4. **SFT 冷启动**
   - 用少量 relative 格式的 `(state → tool call)` 样本教会格式，再 GRPO。

本轮结果说明：在 0.6B 上直接切换到 signed-delta 相对坐标不可行，不能只用 GRPO 硬训。
