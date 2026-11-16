# 单步实验（Phase 1）

目标：不依赖 SFT，用单步 GRPO 验证 Qwen3-0.6B 能否学会最基础的
「当前位置/终点 → 移动方向」接地能力。第一阶段只跑 **L0 空地**。

---

## 1. 数据与 eval

### 训练状态

- 从 `data/maps/train.jsonl` 按 `--levels` 过滤地图；
- 每次 update 采样 `--states` 张地图，每张图用固定 RNG 选一个
  `distance >= 3` 的自由格作为单步状态；
- 同一 update 内尽量不重复 `(map, pos)`。

### 固定 eval

- 优先使用 `data/eval_states.jsonl` 中对应 level 的状态；
- 不够时从 `data/maps/test_id.jsonl` 的对应 level 地图固定补齐；
- 默认 `--eval-states 64`；L0 目前是 60 条 eval_states + 4 条补齐；
- eval 集合会写到：

```
out/phase1_eval_L0_64.json
```

后续 run 直接复用，保证同一组状态可比。要强制重建，删掉该文件。
后续全量实验用 `--eval-states 300`（使用完整 eval_states）。

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

- `valid_rate`：非 `invalid` 的比例；
- `optimal_rate`：`optimal` 比例；
- `side_rate` / `farther_rate` / `blocked_rate` / `invalid_rate`；
- `unique_actions`：每组成员里不同合法方向数的均值；
- `degenerate_rate`：组内奖励无区分度、被跳过的组比例；
- `mean_reward`、`ratio_min/max`、`clip_frac`、`drop_frac`、`grad_norm`；
- 固定 eval 的 `eval_optimal_rate / eval_valid_rate / eval_farther_rate`。

---

## 3. 启动

```bash
conda activate minimind

# 终端 A：UI
python -m env.ui --port 7861

# 终端 B：单步实验（L0）
python scripts/train_phase1.py \
  --levels L0 \
  --updates 100 \
  --states 8 \
  --g 8 \
  --temperature 1.2 \
  --max-tokens 40 \
  --eval-states 64 \
  --eval-every 5 \
  --target-optimal 0.95 \
  --ui-url http://127.0.0.1:7861
```

`--target-optimal 0.95` 表示 eval optimal 达到 95% 就提前停止并保留
`best.pt`；设为 `0` 可关闭早停。L0 通常在 5 个 update 左右饱和，
继续更新大多会变成“全 optimal 退化组”，没有梯度信号。

不接 UI 冒烟：

```bash
python scripts/train_phase1.py --updates 1 --states 2 --g 8 \
  --eval-states 64 --eval-every 1 --no-ui
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
- 对比同一状态下 8 条采样的 action、reward、logprob、latency。

训练看板的「单步指标」卡片会同时显示：

- 本 update 的 optimal / side / farther / blocked / invalid；
- 组内 unique actions、退化组比例；
- 固定 eval 的 optimal / valid / farther 曲线。

---

## 5. 产物

默认：

```
out/phase1/
├── ckpt_u0010.pt
├── ckpt_u0020.pt
├── best.pt                 # eval_optimal_rate 最高
└── last.pt
out/lora_phase1.pt           # 兼容旧脚本的最终 adapter
out/phase1.jsonl             # 每个 update 一行指标
out/phase1_log.json          # 完整 hist
out/phase1_eval_L0_64.json   # 固定 eval 集合
```

---

## 6. 观察与调参

- `valid_rate` 应接近 1；如果掉，先看 `invalid` 和 `max_tokens`。
- `optimal_rate` 应上升；`farther` / `side` 应下降。
- `degenerate_rate` 长期很高：升温度、增加 state 数，或降低组内重复。
- `ratio` 长期贴着 clip：检查 `max_logp_delta`（先 0.5，必要时 0.2）。
- L0 都学不会：先确认固定 eval 的 baseline，再考虑 SFT 冷启动。
