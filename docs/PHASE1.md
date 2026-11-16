# 单步实验（Phase 1）

目标：不依赖 SFT，用单步 GRPO 验证 Qwen3-0.6B 能否学会最基础的
「当前位置/终点 → 移动方向」接地能力。第一阶段只跑 **L0 空地**。

---

## 1. 数据

### 动作均衡状态集

不再每次从地图上随机抽一个位置，而是从 `data/states/` 读取预先构造好的状态集：

```
data/states/
├── train.jsonl    # 训练
├── val.jsonl      # 早停 / best checkpoint
├── test.jsonl     # 最终报告
└── manifest.json
```

每个状态是：

```
map_id + pos + optimal_actions + action_class
```

`optimal_actions` 由 BFS 距离场计算：所有能让距离减 1 的动作。

动作类别：

| class | 含义 |
|---|---|
| `singleton_up/down/left/right` | 只有一个最优方向 |
| `mixed_dr/dl/ur/ul` | 两个最优方向（斜对角） |

L0 配额：

| split | 每个 singleton 方向 | 每个 mixed 组合 | 总数 |
|---|---:|---:|---:|
| train | 64 | 16 | 320 |
| val | 12 | 4 | 64 |
| test | 12 | 4 | 64 |

这样四个方向的 singleton 数量完全相等，同时保留斜对角状态。

### 构建与校验

```bash
python scripts/build_states.py
python scripts/check_states.py
```

`build_states.py` 不重新生成地图，只枚举现有 `data/maps/*.jsonl` 的自由格。
`check_states.py` 会校验 BFS 动作、action_class、配额、四方向均衡和
train/val/test 的 map 不重叠。

> 当前 `/data` 盘可能没有空间，可以把状态集生成到
> `--out /tmp/opencode/data_states`，训练时用
> `--state-dir /tmp/opencode/data_states`。

### natural eval

`data/eval_states.jsonl` 保留为 **natural eval**：

- 不做动作均衡；
- 用于和旧结果对比；
- L0 目前 60 条，原先的 baseline optimal 是 81.25%。

---

## 2. 指标

每个采样动作落入五类之一：

| kind | 含义 |
|---|---|
| `optimal` | BFS 距离减 1 |
| `side` | 距离不变 |
| `farther` | 距离变大 |
| `blocked` | 撞墙/越界 |
| `invalid` | 工具调用格式非法 |

关键指标：

- `val_optimal_rate`：val 集（用于早停）
- `test_optimal_rate`：balanced test（主报告）
- `natural_optimal_rate`：natural eval（对比）
- `optimal / side / farther / blocked / invalid`
- `singleton_accuracy.up/down/left/right`：每个方向单最优状态的正确率
- `mixed_optimal_rate`：斜对角双最优状态的正确率
- `unique_actions`、`degenerate_rate`
- `mean_reward`、`ratio_min/max`、`clip_frac`、`drop_frac`、`grad_norm`

balanced test 会强制包含 down-only 状态，所以如果模型仍然不会 `down`，
这个指标会直接暴露，而不是被斜对角双最优状态掩盖。

---

## 3. 启动

```bash
conda activate minimind

# 终端 A：UI
python -m env.ui --port 7861

# 终端 B：单步实验（L0）
python scripts/train_phase1.py \
  --levels L0 \
  --updates 20 \
  --states 16 \
  --g 8 \
  --lr 5e-5 \
  --temperature 1.2 \
  --max-tokens 40 \
  --eval-every 5 \
  --target-optimal 0.90 \
  --ui-url http://127.0.0.1:7861
```

`--target-optimal 0.90` 表示 val optimal 达到 90% 就提前停止并保留
`best.pt`；设为 `0` 可关闭早停。

不接 UI 冒烟：

```bash
python scripts/train_phase1.py --updates 1 --states 2 --g 8 \
  --eval-every 1 --no-ui
```

远程访问：

```bash
python -m env.ui --host 0.0.0.0 --port 7861
```

---

## 4. UI 怎么看 8 条采样

每个 update 的每个 state 的 G 条采样都会生成独立 episode：

```
u{update}s{state_index}g{sample_index}
```

例如 update 1、state 0 的采样是 `u1s0g0` … `u1s0g7`。

- 打开多个浏览器窗口/标签页，都连同一个 UI；
- 每个窗口在「推理轨迹」里 pin 不同 episode；
- 在「筛选 episode」输入 `u1s0`，只显示该状态的 8 条采样；
- 对比同一状态下 8 条采样的 action、reward、logprob、latency。

训练看板首屏显示：

- balanced test / natural test / val 的 optimal；
- degenerate rate、valid rate、down singleton accuracy；
- 四个方向的 singleton accuracy 条形图；
- val / balanced / natural 的 optimal 曲线。

其他训练诊断（loss、ratio、clip、drop、farther、blocked 等）折叠在
「训练诊断 / 全部指标」里。

---

## 5. 产物

默认：

```
out/phase1/
├── ckpt_u0005.pt
├── ckpt_u0010.pt
├── best.pt                 # val_optimal_rate