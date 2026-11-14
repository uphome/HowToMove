# 数据集生成指南

> 目标读者：要重新生成 / 调整 / 扩展 `data/` 的人。
> 相关文件：`dataset.py`、`scripts/build_dataset.py`、`scripts/check_dataset.py`、
> `scripts/render_dataset.py`。生成结果的分析见 `reports/dataset.md`。

---

## 1. 一次性生成（TL;DR）

```bash
cd E:/codeall/AgenticRL
python scripts/build_dataset.py      # 生成 data/，约 18 秒
python scripts/check_dataset.py      # 门禁，必须 PASS
python scripts/render_dataset.py     # 预览图 → out/ds_*.png
```

产物约 1.2 MB：

```
data/
├── manifest.json
├── eval_states.jsonl        300
└── maps/
    ├── train.jsonl         3000
    ├── val.jsonl            400
    ├── test_id.jsonl        400
    ├── test_hard.jsonl      200
    └── test_ood.jsonl       200
```

不需要模型、不需要 GPU、不需要 torch。纯标准库 + `env/` 的生成器。

---

## 2. 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.10+（只用标准库） |
| 目录 | 在项目根目录执行（`dataset.py` / `env/` 需在 `sys.path`） |
| 依赖 | 无。`scripts/*.py` 会自动把项目根插入 `sys.path` |
| 时间 | 默认规模约 18 秒；生成 6000 seeds/级 |

---

## 3. 生成流程

```
① 造图        每个级别 0..N-1 个 seed 调 generate()，得到候选地图
② 去重        按 (w,h,walls,start,goal) 的规范化 JSON 求 sha256
③ 分桶        按哈希前 8 位 % 100 落桶，决定 split
④ 配额        每个 (split, level) 按哈希排序取前 K
⑤ OOD         8x8 / 16x16 单独生成，只取 test_hard 桶
⑥ 评测状态    从 test_id 地图里抽固定状态，重算 optimal_actions
⑦ 落盘        写 jsonl + manifest（含每个文件的 sha256）
```

### 3.1 造图：复用 `env/maze.py::generate`

```python
from env.maze import generate

maze = generate(level="L3", width=12, height=12, max_steps=30, seed=0)
```

内部三步：

1. `_make_walls(level, rng)` 生成形状
   - L0 空地
   - L1 竖墙 + 1 格门
   - L2 L 型墙
   - L3 U/C 型墙（窄门）
   - L4 随机 1~3 个形状组合
2. `_place(...)` 随机定终点 → 反向 BFS → 扫描**全部**自由格，只保留满足
   `min_dist=8`、`dist <= max_steps - 2 = 28`、`detour >= 级别要求` 的格子，
   再随机取一个当起点。单次扫描、不盲试。
3. 凑不出合法图就换一组墙重试，上限 200 次；仍失败则抛 `RuntimeError`，
   构建脚本会跳过并计数。

级别约束（`env/maze.py::LEVEL_RULES`）：

| 级别 | min_dist | min_detour | 含义 |
|---|---|---|---|
| L0 | 8 | 0 | `detour` 恒为 0 |
| L1 | 8 | 2 | 至少一步"远离"目标 |
| L2 | 8 | 2 | 同上 |
| L3 | 8 | 4 | 至少两步远离（U 型陷阱） |
| L4 | 8 | 2 | 组合形状 |

> `detour = shortest - manhattan` 在 4 邻域网格上**恒为偶数**
> （每多走一步远离就多 2），所以只能取 2/4，不能取 1/3。

### 3.2 去重：规范化 JSON

```python
canonical = json.dumps(
    {"w": 12, "h": 12,
     "walls": sorted([x, y] for x, y in walls),
     "start": [x, y], "goal": [x, y]},
    sort_keys=True, separators=(",", ":"),
)
h = hashlib.sha256(canonical.encode()).hexdigest()
```

**必须**排序 `walls`、固定分隔符，否则同一张图会算出不同哈希。

为什么必须去重：L0 只有起点/终点两个变量，2000 个 seed 里只有 1798 张唯一图
（重复率 10%）。不去重会把同一张图切进 train 和 test。

### 3.3 分桶：内容哈希决定 split

```python
bucket = int(h[:8], 16) % 100
```

| bucket | split | 用途 |
|---|---|---|
| 0–69 | `train` | RL rollout 采样 |
| 70–83 | `val` | 训练中定期评估 / 选超参 |
| 84–91 | `test_id` | 同分布最终指标 |
| 92–99 | `test_hard` | 只收 `detour >= 6` 且级别 ∈ L2/L3/L4 |

同一张图的哈希相同 → **必然落在同一 split**，结构上杜绝泄漏。
bucket 只决定归属，数量由配额决定。

### 3.4 配额：按哈希排序取前 K

```python
pool = sorted(candidates[split][level], key=lambda r: r.hash)
picked = pool[:k]
```

按哈希排序而不是 seed 排序，避免生成顺序带来的偏置。
配额不足会打印 `!! ... 不足`，但不会中断；`check_dataset.py` 会把它判为失败。

### 3.5 OOD：换尺寸

```python
OOD_SIZES = ((8, 8), (16, 16))
```

每个尺寸、每个级别（L1–L4）用 `--ood-seeds` 个 seed 生成，只从 `test_hard`
桶取，并排除主集已出现的哈希。作用是检验模型是否只是"记住了 12×12"。

### 3.6 单步评测状态

从 `test_id` 的地图里每级取 `eval_states // 5` 张，每张随机选一个
`dist >= 3` 的自由格：

```json
{"id":"es-0000","map_id":"L3-000123","level":"L3","pos":[6,1],
 "state":"pos=(6,1) goal=(10,7)\nwalls: (3,0) ...",
 "optimal_actions":["down"],"distance":14,"manhattan":10}
```

`optimal_actions` 由 BFS 重算：所有能把距离减 1 的动作（开阔区域可能并列）。
**它只用于算单步准确率，不参与训练。**

---

## 4. 构建参数

```bash
python scripts/build_dataset.py --help
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--out` | `data` | 输出目录 |
| `--seeds` | 6000 | 主集每个级别的候选 seed 数 |
| `--ood-seeds` | 2500 | 每个尺寸 × 级别的 OOD 候选数 |
| `--train-per-level` | 600 | train 每级配额 |
| `--val-per-level` | 80 | val 每级配额 |
| `--test-per-level` | 80 | test_id 每级配额 |
| `--hard` | 200 | test_hard 总数 |
| `--ood-per-level` | 25 | OOD 每个尺寸每级的配额（总数 = ×4 级 ×2 尺寸） |
| `--eval-states` | 300 | 单步评测状态总数（均分到 5 个级别） |
| `--max-steps` | 30 | 写进每条记录的步数预算 |

小规模冒烟：

```bash
python scripts/build_dataset.py --out data_small \
    --seeds 1500 --ood-seeds 600 \
    --train-per-level 100 --val-per-level 20 --test-per-level 20 \
    --hard 40 --ood-per-level 5 --eval-states 50
```

`check_dataset.py` 以 `manifest.json` 里的配额为准，所以任意规模都能校验。

---

## 5. 输出格式

### 5.1 地图记录（`maps/*.jsonl`）

一行一条：

```json
{"id":"L3-000123","level":"L3","seed":123,"w":12,"h":12,"max_steps":30,
 "walls":[[2,8],[5,8]],"start":[4,9],"goal":[10,11],
 "shortest":12,"detour":4,"hash":"a1b2c3d4..."}
```

| 字段 | 说明 |
|---|---|
| `id` | 主集 `{level}-{seed:06d}`；OOD `ood{尺寸}-{level}-{hash[:8]}` |
| `seed` | 生成时的 seed（仅追溯用，重建不依赖它） |
| `walls` | 显式坐标，与生成器版本解耦 |
| `shortest` | BFS 最短路长度 |
| `detour` | `shortest - manhattan`，恒为偶数 |
| `hash` | 规范化 JSON 的 sha256，跨文件唯一 |

**BFS 距离场不进文件**，加载时用 `bfs_field` 重算（O(144)）。

### 5.2 `manifest.json`

```json
{
  "version": 1,
  "created_at": "2026-11-14T15:47:58",
  "builder": "scripts/build_dataset.py",
  "generator": {"file": "env/maze.py", "level_rules": {...},
                "max_steps": 30, "main_size": [12,12],
                "ood_sizes": [[8,8],[16,16]]},
  "split_rule": "sha256(canonical)[:8] % 100 -> train:<70 val:<84 test_id:<92 test_hard:<100",
  "seeds": {"main": 6000, "ood": 2500},
  "quotas": {...},
  "files": {"maps/train.jsonl": {"count": 3000, "sha256": "..."}},
  "counts": {"maps/train.jsonl": 3000}
}
```

---

## 6. 校验

```bash
python scripts/check_dataset.py            # 默认 data/
python scripts/check_dataset.py --data data_small
```

10 项检查：

1. manifest 里每个文件的 sha256 与实物一致；
2. 五个 split 的 `hash` 两两无交集；
3. 每张图可解（`dist != INF`）；
4. `shortest <= max_steps - 2`（否则终局奖励永远拿不到）；
5. `detour == shortest - manhattan`；
6. `hash` 与规范化 JSON 重算一致；
7. 起点/终点不在墙上；id 在本 split 内唯一、无重复地图；
8. `test_hard` 的 `detour >= 6` 且级别 ∈ L2/L3/L4；
9. `eval_states` 的 `optimal_actions`、`state` 文本与 BFS/渲染重算一致；
10. 各级别配额达标（以 manifest 的 `quotas` 为准）。

任何一项失败都返回非零退出码，可直接接 CI。

---

## 7. 在代码里使用

```python
from dataset import MapDataset

train = MapDataset.load("data/maps/train.jsonl")
print(train.levels())                 # {'L0': 600, ..., 'L4': 600}

import random
rng = random.Random(0)
picks = train.sample(4, rng)          # 4 条 MapRecord
maze = picks[0].to_maze()             # 重建 Maze（含 BFS 距离场）

hard = MapDataset.load("data/maps/test_hard.jsonl").filter(min_detour=6)
```

单步评测状态：

```python
import json
states = [json.loads(l) for l in open("data/eval_states.jsonl", encoding="utf-8")]
for s in states:
    # s["state"] 直接作为观测文本；s["optimal_actions"] 用于判分
    ...
```

配对观测与 prompt：

```python
from env.render import render_state
from prompts import render_prompt

state_text = render_state(maze, maze.start)        # 默认 minimal 风格
prompt_text, prompt_ids = render_prompt(tokenizer, state_text)
```

---

## 8. 修改数据集

### 8.1 改规模

只调配额即可，候选池有 2.8 万张，远大于需求：

```bash
python scripts/build_dataset.py --train-per-level 1000 --val-per-level 100 \
    --test-per-level 100 --hard 300
```

### 8.2 改分桶比例

编辑 `dataset.py`：

```python
SPLIT_BUCKETS = (
    (0, 70, "train"),
    (70, 84, "val"),
    (84, 92, "test_id"),
    (92, 100, "test_hard"),
)
```

改完必须重新生成 + 校验；**旧数据集不能和新规则混用**（哈希归属会变）。

### 8.3 改尺寸

`scripts/build_dataset.py` 顶部的常量：

```python
MAIN_SIZE = 12
OOD_SIZES = ((8, 8), (16, 16))
```

`generate()` 本身支持任意 `width/height`，但 `LEVEL_RULES` 的 `min_dist=8`
在过小的网格上会收紧可选范围（例如 6×6 几乎没有候选）。

### 8.4 加新级别

1. `env/maze.py`：加 `LEVELS`、`LEVEL_RULES`、`_SHAPES`；
2. 重新生成数据集；
3. `check_dataset.py` 会自动按 `LEVELS` 校验配额。

⚠️ 改生成器会改变地图内容，**旧数据集的 `hash` 与描述不再对应**，必须重建。

---

## 9. 确定性

- 同一台机器、同一份 `env/maze.py`、同样的参数 → **字节级相同**的输出
  （已实测两次构建的 sha256 一致）；
- 输入是 `seed = 0..N-1`，与时钟、随机源无关；
- `eval_states` 用固定 RNG seed（`build_eval_states(..., seed=0)`）；
- manifest 记录了 `level_rules`、尺寸、seed 范围、配额，便于追溯。

---

## 10. 常见问题

**Q: 某个 `split/level` 报"不足"？**

说明该桶里合格地图不够配额。按比例加 `--seeds`（不足数量的 5 倍左右），
或下调该级配额。`check_dataset.py` 会把它判为失败。

**Q: `check_dataset.py` 报 sha256 不匹配？**

文件被手工改过，或构建中断。重新跑 `build_dataset.py`。

**Q: 能把 `data/` 提交到仓库吗？**

可以，1.2 MB。若只想提交生成规则，把 `data/` 加进 `.gitignore`，并在
CI 里跑 `build_dataset.py && check_dataset.py`。

**Q: 训练时能直接用 `(level, seed)` 现生成吗？**

不建议。生成器一改，训练/评测的数据分布就悄悄变了，指标不可比。
用冻结的地图集，生成器只影响"以后新造的数据"。
