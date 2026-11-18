# Layout / Task 数据集报告

> 日期：2026-11-18 ｜ 版本：v2 ｜ 状态：**PASS**
> Builder：`scripts/build_layout_tasks.py`
> 数据目录：`data/layouts/`、`data/tasks/`、`data/layout_tasks_manifest.json`

---

## 1. 结论

当前 Agent SFT/评估使用的数据已经从旧的“一张图 + 一个 start/goal”
改成了：

```text
Layout（墙形） → Task（start/goal） → Trajectory（按需生成） → Turn
```

当前规模：

| split | layouts | tasks | L0 | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 481 | 3000 | 600 | 600 | 600 | 600 | 600 |
| val | 65 | 400 | 80 | 80 | 80 | 80 | 80 |
| test | 65 | 400 | 80 | 80 | 80 | 80 | 80 |

- L1–L4 的 layout 在 train/val/test 之间完全不相交；
- 所有 split 的 task_id 完全不相交；
- 每个 task 都可解，且 `shortest <= max_steps - 2`。

`data/states/` 和旧的单步 `eval_states.jsonl` 已移除；轨迹不落盘，
由 `agent/sft_data.py` 从 tasks 按需生成。

---

## 2. 数据单位

### 2.1 Layout

Layout 只描述静态墙形：

```json
{
  "layout_id": "...",
  "level": "L1",
  "width": 12,
  "height": 12,
  "max_steps": 30,
  "walls": [[0,4], [1,4], ...],
  "source_id": "..."
}
```

同一条墙形可以对应多个 start/goal，也就是多个 Task。

### 2.2 Task

Task 在 layout 上增加起点、终点和 BFS 属性：

```json
{
  "task_id": "...",
  "layout_id": "...",
  "level": "L1",
  "start": [4, 7],
  "goal": [2, 2],
  "shortest": 9,
  "detour": 2,
  "hash": "..."
}
```

### 2.3 Trajectory

Trajectory 不存入 `data/`：

```text
Task → BFS shortest path → SFT turn samples
```

每次训练时由 `agent/sft_data.py` 根据 task 和随机 tie-break 生成。

### 2.4 训练时恢复样本（不落盘）

Layout/Task 数据集本身不包含撞墙恢复动作。SFT 训练时可以通过
`--recovery-per-task N` 动态生成恢复样本：

```text
最优路径状态
→ 注入一个确定会撞墙/撞边界的动作
→ 生成 blocked=true 的 tool result
→ target 是该状态下的正确最优动作
```

示例：

```text
assistant: move(up)
tool: {"moved":"up","ok":false,"blocked":true,
       "blocked_by":"wall","reason":"You hit a wall..."}

assistant: move(right)   ← 训练目标
```

恢复样本：

- 只用于 train split；
- 不写入 `data/tasks/`；
- 不加入 val/test；
- 由 `agent/sft_data.py` 在构建 SFT items 时生成；
- 每个 task 最多注入 N 个恢复状态。

当前 L1 实验的 SFT turn 数：

```text
无 recovery:        7103
recovery=2:         8247
source maps + r=2:  7819
source maps + r=8:  9154
```

#### 2.4.1 可复现性

恢复样本虽然不落盘，但**不是无种子的随机生成**。每个 task 使用：

```python
task_seed = (int(rec.hash[:16], 16) ^ (seed * 1000003)) & 0xFFFFFFFF
rng = random.Random(task_seed)
```

以下内容均由这个 RNG 决定：

- 最短路径的随机 tie-break；
- 哪些路径状态注入恢复样本；
- 使用哪个撞墙方向。

task 选择先按 task hash 排序，再取前 N 个，因此也不依赖文件遍历顺序。

固定配置下，相同 seed 生成相同样本。下面的 fingerprint 对每个
SFT item 的 `map_id`、`turn`、`action`、`prompt_ids`、`completion_ids`
依次做 sha256，再取前 24 位。

使用当前 `data/tasks/train` 的 600 个 L1 task、`recovery_per_task=2`、
`history_window=12`、`max_seq_len=4096` 做验证：

```text
seed=7 第一次:
  items=8247
  fingerprint=72f5bf5f3379c5ee6e38c89a

seed=7 第二次:
  items=8247
  fingerprint=72f5bf5f3379c5ee6e38c89a

seed=8:
  items=8242
  fingerprint=81bd2220b7e438c3ca787c86
```

因此：

```text
相同 seed + 相同 task 数据 + 相同代码/配置
→ 恢复样本相同
```

但以下变化会改变生成结果：

- 修改 `--seed`；
- 修改 `--recovery-per-task`；
- 修改训练 task 集合或 `train_source`；
- 修改 `agent/sft_data.py` 的分桶/路径/恢复逻辑；
- 修改 prompt 模板或 tokenizer 版本。

所以它是“固定代码版本下可复现”，不是跨代码版本永久不变。
如需跨版本审计，建议在训练 JSON/checkpoint 中额外保存：

```text
sft_items_sha256
recovery_per_task
seed
train_source
dataset_manifest_sha256
```

---

## 3. Split 规则

| 级别 | split 规则 | 说明 |
|---|---|---|
| L0 | task-level | 只有一个空地 layout，按 task hash 分 train/val/test |
| L1–L4 | layout-level | 同一 layout 的所有 task 只能落在同一个 split |

L1–L4 的配额：

```text
每个 level:
  train: 120 layouts × 5 tasks = 600 tasks
  val:    16 layouts × 5 tasks =  80 tasks
  test:   16 layouts × 5 tasks =  80 tasks
```

另外加上 L0 的 600/80/80 个 task。

这样布局级泄漏被结构性地阻止：val/test 的墙形不会出现在 train 中。

---

## 4. 文件

```text
data/
├── layouts/
│   ├── train.jsonl     481 layouts
│   ├── val.jsonl        65 layouts
│   └── test.jsonl       65 layouts
├── tasks/
│   ├── train.jsonl    3000 tasks
│   ├── val.jsonl       400 tasks
│   └── test.jsonl      400 tasks
├── layout_tasks_manifest.json
└── maps/                   旧 v1 源地图，保留作为来源/历史评估
```

`data/layout_tasks_manifest.json` 记录：

- builder 和 split 规则；
- 各 split 的 layout/task 数量；
- 每个数据文件的 sha256 和行数。

---

## 5. 生成与校验

重新生成：

```bash
python scripts/build_layout_tasks.py --data data --out data
```

校验 layout/task：

```bash
python scripts/check_layout_tasks.py data
```

当前结果：

```text
train layouts 481 tasks 3000
val   layouts  65 tasks  400
test  layouts  65 tasks  400
PASS layout/task checks
```

校验内容：

- manifest 的 sha256 和行数；
- L1–L4 layout 两两不重叠；
- task_id 两两不重叠；
- start/goal 不越界、不在墙上；
- BFS 可解，且 `shortest`、`detour` 与重算结果一致；
- `shortest <= max_steps - 2`；
- 满足各级别的 `min_dist` / `min_detour` 约束。

旧 map-only 数据集仍可通过：

```bash
python scripts/check_dataset.py --data data
```

结果：

```text
PASS  数据集检查全部通过
地图总数 4200
```

---

## 6. 已知限制

- 当前仍是固定 12×12；任意尺寸尚未实现；
- L0 只有一个空地 layout，L0 的泛化是 task-level 泛化；
- L1–L4 每级只选了 120 个 train layout；
- 源 `data/maps/train` 中实际有更多唯一 layout（例如 L1 有 532 个），
  后续 SFT 实验使用 `--train-source maps` 来扩大 layout 覆盖；
- 每个 layout 只生成 5 个 task，start/goal 覆盖仍可继续扩展。
