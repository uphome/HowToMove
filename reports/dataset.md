# 数据集报告

> 日期：2026-11-14 ｜ 版本 v1 ｜ 状态：**已生成并通过门禁**

---

## 1. 结论

`data/` 已生成 4200 张地图 + 300 个单步评测状态，三个 split 的内容哈希两两不重叠，
所有地图可解且 `shortest <= max_steps-2`，`scripts/check_dataset.py` 全绿。

```
maps/train.jsonl        3000   600/级     929e529ad3b2
maps/val.jsonl           400    80/级     c5d4470e8c0f
maps/test_id.jsonl       400    80/级     966cfd5d1360
maps/test_hard.jsonl     200  detour≥6   8ffd65d132ed
maps/test_ood.jsonl      200  8x8/16x16  91beb2a0b874
eval_states.jsonl        300  只评测      ec5e61a8cf75
```

总体积 1.2MB，生成耗时 16.5s。

---

## 2. 生成流程

### 2.1 造图

复用 `env/maze.py::generate(level, seed)`：

1. `_make_walls`：L0 空地 / L1 竖墙+门 / L2 L 型 / L3 U 型窄门 / L4 随机 1~3 个；
2. `_place`：随机定终点 → 反向 BFS → 扫全部自由格，筛掉不满足
   `min_dist=8`、`dist<=max_steps-2=28`、`detour>=级别要求` 的格，再随机取起点；
3. 不满足就换一组墙重试（上限 200 次）。

主集每级 6000 个 seed，共 30000 次生成 → **28360 张唯一地图**，失败 0。

### 2.2 去重

以

```json
{"w":12,"h":12,"walls":[[x,y],...],"start":[x,y],"goal":[x,y]}
```

的**规范化 JSON**（walls 排序、固定分隔符）做 sha256。

> 为什么必须去重：L0 只有起点/终点两个变量，2000 个 seed 里只有 1798 张唯一图
> （重复率 10%）。不去重会把同一张图切进 train 和 test。

### 2.3 分桶

```
bucket = int(sha256(canonical)[:8], 16) % 100

  0..69   train
 70..83   val
 84..91   test_id
 92..99   test_hard（只收 detour >= 6 且级别 ∈ L2/L3/L4）
```

内容相同的图哈希相同 → 必然落在同一 split，**结构上杜绝泄漏**。

分桶后的候选池：train 19786 / val 4020 / test_id 2318 / test_hard 2236，
按配额取前 K（按 hash 排序，避免 seed 顺序偏置）。

### 2.4 OOD

8×8 与 16×16 各生成 2500 seeds/级（L1–L4），同样只从 bucket 92–99 取，
并排除已出现在主集里的哈希。各 100 张，按级别轮转取。

### 2.5 eval_states

从 `test_id` 的 400 张图里每级取 60 张，每张随机选一个 `dist>=3` 的状态，
写入：

```json
{"id":"es-0000","map_id":"L3-000123","level":"L3","pos":[6,1],
 "state":"pos=(6,1) goal=(10,7)\nwalls: (3,0) ...",
 "optimal_actions":["down"],
 "distance":14,"manhattan":10}
```

`optimal_actions` 由 BFS 重算（开阔区域可能有并列）。它**只用于算单步 acc，
不参与训练**。

---

## 3. 数据集分布

| 级别 | train | val | test_id | test_hard |
|---|---|---|---|---|
| L0 | 600 | 80 | 80 | 0 |
| L1 | 600 | 80 | 80 | 0 |
| L2 | 600 | 80 | 80 | 67 |
| L3 | 600 | 80 | 80 | 67 |
| L4 | 600 | 80 | 80 | 66 |

`detour >= 6` 的占比：L2 14%、L3 32%、L4 20% —— hard 集按级别轮转，接近均分。

---

## 4. 门禁（`scripts/check_dataset.py`）

| # | 检查 | 结果 |
|---|---|---|
| 1 | manifest sha256 与文件一致 | ✅ |
| 2 | 五个 split 的 hash 两两不相交 | ✅ |
| 3 | 每张图可解、`shortest <= max_steps-2` | ✅ |
| 4 | `detour == shortest - manhattan` | ✅ |
| 5 | `hash` 与规范化 JSON 重算一致 | ✅ |
| 6 | 起点/终点不在墙上；id 唯一 | ✅ |
| 7 | `test_hard` 的 `detour >= 6` 且级别正确 | ✅ |
| 8 | `eval_states` 的 `optimal_actions` 与 BFS 一致 | ✅ |
| 9 | `eval_states.state` 与 `render_state` 一致 | ✅ |
| 10 | 配额达标（每级 600/80/80） | ✅ |

---

## 5. 预览

| 文件 | 内容 |
|---|---|
| `out/ds_levels.png` | L0–L4 各一张，带最短路 |
| `out/ds_L3.png` | 6 张 U 型墙（agent 被关在箱子里） |
| `out/ds_hard.png` | 6 张 `detour>=6` |
| `out/ds_ood_8.png` / `ds_ood_16.png` | 8×8 / 16×16 |
| `out/ds_sample.png` | 单张，34px/格 |

模型实际看到的观测（L3 样例，13 面墙）：

```
pos=(6,1) goal=(10,7)
walls: (3,0) (4,0) (5,0) (6,0) (7,0) (3,1) (7,1) (3,2) (7,2) (3,3) (5,3) (6,3) (7,3)
```

---

## 6. 已知限制

| 项 | 说明 |
|---|---|
| L0 重复率高 | 靠哈希去重解决；配额仍能满足 |
| `test_hard` 不含 L0/L1 | 这两级结构上 `detour` 很小，硬塞会名不副实 |
| OOD 未调 `min_dist` | 8×8 上 `min_dist=8` 接近上限，选出的图偏"对角线" |
| 未做难度分层采样 | 目前是每级等量；等课程控制器做好再按 detour 分档 |

---

## 7. 复现

```bash
python scripts/build_dataset.py          # 生成 data/
python scripts/check_dataset.py          # 门禁
python scripts/render_dataset.py         # 预览图 → out/
python -m pytest -q                      # 73 passed
```
