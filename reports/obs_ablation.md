# 观测编码消融报告

> 日期：2026-11-14 ｜ 触发问题：「是不是提示词不对？试试只给起点、终点、障碍」
> 结论：**不是提示词问题。Qwen3-0.6B 在当前阶段对观测是"恒等映射"——输出与输入无关。**

---

## 1. 实验设置

- 模型：Qwen3-0.6B（Instruct），greedy（`temperature=0`）
- 样本：30 个状态，L0–L4 每级 6 个，位置随机取「距目标 ≥ 3 的自由格」
- 提示：`prompts.render_prompt`（同一份 system + tools + chat template）
- 指标：`optimal`（选到 BFS 距离 −1 的动作）、`closer`（距离变小）、
  `blocked`（撞墙/越界）、`farther`（变远）、`invalid`（格式非法）

四种编码：

| 风格 | 形态 | prompt 长度 |
|---|---|---|
| `grid` | 12×12 ASCII 网格 + `A`/`G` | ~290 tok |
| `coords` | 坐标 + 障碍坐标列表（项目书原始设计） | ~287 tok |
| `rows` | 坐标 + 障碍按行分组、x 连续段压缩 | **~276 tok** |
| `bitmap` | 坐标 + 12 行 0/1 占用图 | ~405 tok |

---

## 2. 结果

`history=0`（不带上一步）：

| 风格 | valid | optimal | closer | blocked | farther | invalid |
|---|---|---|---|---|---|---|
| grid | 30/30 | 14 | 0 | 3 | 13 | 0 |
| coords | 30/30 | 14 | 0 | 3 | 13 | 0 |
| rows | 30/30 | 14 | 0 | 3 | 13 | 0 |
| bitmap | 30/30 | 14 | 0 | 3 | 13 | 0 |

**四个风格的 30 个预测完全一样，全是 `right`。**

14/30 的 `optimal` 不是"学会了"，而是"常量策略 `right` 恰好在这 14 个状态上
缩短了距离"——等价于随机基线。

`history=3` 结果完全相同（全 `right`）。

---

## 3. 对照：模型到底能不能听懂

同一套 tools + chat template，换掉 user 内容：

| user 内容 | 模型输出 | 判定 |
|---|---|---|
| `Use the move tool with direction left.` | `{"direction": "left"}` | ✅ 完全正确 |
| `Use the move tool with direction up.` | `{"direction": "up"}` | ✅ 完全正确 |
| `pos=(9,3) goal=(4,2) delta=(-5,-1). You must move left. Use the move tool.` | `{"direction": "left"}` | ✅ |
| `pos=(9,3) goal=(4,2) delta=(-5,-1). You must move up. Use the move tool.` | `{"direction": "up"}` | ✅ |
| `pos=(9,3) goal=(4,2) delta=(-5,-1). Which way? Use the move tool.` | `The agent should move **up** to reach the goal.`（不给工具调用） | ❌ |
| 完整迷宫观测 | `{"direction": "right"}` | ❌ |

**结论**：

1. 工具调用格式、schema、模板**全部正确** —— 只要告诉它方向，它 100% 正确输出；
2. 它也能**部分**读到信息（纯文字问句时答出了 `up`，对应 `dy=-1`）；
3. 但它**不会**自己做「delta → 方向」这个映射，也不会在不确定时调用工具，
   而是退化成常量 `right`。

这是**能力/接地缺口**，不是编码问题。换 4 种编码、去掉历史、换顺序，输出一模一样。

---

## 4. 对"模型知不知道上一步走哪了"

当前 rollout 的设计是 **环境多步、模型单步**：

```
第 t 轮：把「当前状态」整份重新给出（无对话历史）
   maze / pos / goal / delta / walls
   last=最近 3 个动作        ← 只有这个算"记忆"
```

- 模型**知道自己在哪**（`pos` 每轮更新），这是它"上一步走到哪"的直接答案；
- 但**不知道完整轨迹**，只看到最近 3 个动作，看不到走过的格子；
- 好处：prompt 长度恒定（~290 tok），不会被 30 步历史撑爆（引擎上限 2048）；
- 代价：可能在原地打转，除非 `last` 或 `visited` 能提示它。

已新增两个开关，等 SFT 后验证：

```python
run_episode(..., obs_style="grid", memory=3, show_visited=True)
# show_visited: 走过的格子渲染成 ','
# memory=0:     完全不给 last
```

---

## 5. 结论与行动项

| 项 | 决定 |
|---|---|
| 观测编码 | 保留 4 种风格，**默认仍是 `grid`**；`rows` 最省 token，留作 SFT 后对照 |
| 编码是否是瓶颈 | **不是**。当前模型对任何编码都输出常量 |
| SFT 是否必须 | **必须**，而且第一课不是格式（格式已 100% 正确），是 **delta → 方向** |
| SFT 数据设计 | 先教 `delta=(dx,dy) → direction`（几百条即可），再教绕墙 |
| 历史 | 先保持 `memory=3`；`show_visited` 在 SFT 后的对照实验里评估 |

SFT 之后的验收标准（替代当前 `optimal` 基线）：

- 空地随机状态：`optimal` ≥ 95%
- L1（单墙）：成功率 ≥ 60%
- L3（U 型墙）：成功率 ≥ 20%（留给 RL 提升）

---

## 6. 复现

```bash
conda activate qwen3
cd E:/codeall/AgenticRL

# 四种编码 × 30 状态
python scripts/obs_ablation.py --n 30 --history 0

# 只看某一种编码、比较历史长度
python scripts/obs_ablation.py --n 30 --styles rows --history 0,1,3
```
