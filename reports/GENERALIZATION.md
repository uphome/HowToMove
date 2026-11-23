# 泛化性测试：大小与障碍物

> 日期：2026-11-23 ｜ 模型：最终 GRPO checkpoint
> `/tmp/opencode/grpo_unified_chunk17/grpo_agent.pt`
> 评估：greedy，temperature=0，history_window=12，max_seq_len=4096

---

## 1. 结论

这次做了两组**零样本**测试：

```text
1. 大小泛化：8x8、10x10、12x12、16x16、20x20
2. 同墙不同 start/goal：12x12 的 L4 墙体
```

结论：

```text
大小泛化：比预期好，但明显受 max_steps 预算影响
障碍物泛化：同墙换 start/goal 基本保持；新墙体形状尚未测试
```

固定 `max_steps=30` 时：

```text
8x8  L0 100 / L4 86
10x10 L0 100 / L4 87
12x12 L0  99 / L4 84
16x16 L0  97 / L4 84
20x20 L0  93 / L4 77
```

20×20 使用更长预算后：

```text
20x20, max_steps=80:
L0 98/100
L4 93/100
```

所以 20×20 固定 30 步的明显下降，很大一部分是**步数预算**问题，而不是模型完全无法读取大图。

同墙不同 start/goal：

```text
L4: 68 / 75 = 90.7%
```

对比原始 L4 test：

```text
89 / 100 = 89%
```

差别很小，说明模型不是只记住固定 start/goal。

---

## 2. 大小泛化

### 2.1 固定 max_steps=30

每个尺寸、每个 level 100 张测试图，共 200 张；墙体和 start/goal
由新随机种子生成，未参与训练。

| size | max_steps | L0 | L4 | total |
|---|---:|---:|---:|---:|
| 8×8 | 30 | 100/100 | 86/100 | 186/200 |
| 10×10 | 30 | 100/100 | 87/100 | 187/200 |
| 12×12 | 30 | 99/100 | 84/100 | 183/200 |
| 16×16 | 30 | 97/100 | 84/100 | 181/200 |
| 20×20 | 30 | 93/100 | 77/100 | 170/200 |

12×12 是同一套生成器重新采样的对照，L4 为 84/100；原始标准 test
的 L4 为 89/100，差异来自不同的 held-out 地图，不是模型变化。

### 2.2 延长步数预算

16×16 和 20×20 另外生成了长预算测试集：

```text
16x16: max_steps=64
20x20: max_steps=80
```

| size | max_steps | L0 | L4 | total |
|---|---:|---:|---:|---:|
| 16×16 | 64 | 100/100 | 93/100 | 193/200 |
| 20×20 | 80 | 98/100 | 93/100 | 191/200 |

固定预算和长预算使用不同的地图样本，因此这不是严格的同图预算消融；
但趋势很清楚：大尺寸下，30 步会成为主要瓶颈。

### 2.3 解释

- 8×8–12×12：基本保持，L0 接近满分，L4 仍在 84–87。
- 16×16：固定 30 步仍能保持 L4 84；延长预算后到 93。
- 20×20：固定 30 步降至 L4 77；延长预算后恢复至 93。
- 这说明模型对**尺寸本身**有一定零样本适应，但不能用固定 30 步
  评价大图。
- 长预算下仍有少量格式错误：16×16 为 3 个；需要继续观察。

---

## 3. 同墙不同 start/goal

使用原始 12×12 L4 test 的墙体，保持墙体完全不变，重新选择
start/goal。

```text
wall set:      L4 test walls
maps:          75
level:         L4
max_steps:     30
```

结果：

| 条件 | success | mean steps | collisions | format errors |
|---|---:|---:|---:|---:|
| 原始 L4 test | 89/100 | 17.0 | 127 | 0 |
| 同墙新 start/goal | **68/75 = 90.7%** | 17.8 | 72 | 0 |

同墙换 start/goal 没有明显下降，说明：

```text
模型不是只记住固定的 start/goal 配对
```

但它仍然不能证明新墙体形状的泛化，因为墙体完全来自训练分布中的
L4 形状族。

---

## 4. 仍未测试

```text
1. 全新障碍物形状族
   - 斜墙、房间、窄走廊、封闭区域、复杂多连通墙
2. 新的墙体密度
3. 同一 wall set 的更多 start/goal 组合（当前 75 个）
4. 严格同图、不同 max_steps 的预算消融
5. 训练时加入多尺寸 / OOD 墙形后的对比
```

当前证据可以表述为：

```text
12x12 未见 wall set：有泛化证据
12x12 同墙新 start/goal：有泛化证据
8x8–16x16 零样本：基本可用
20x20 零样本：依赖步数预算
新障碍物形状：尚未测试
```

所以还不能说已经具备“任意尺寸、任意障碍物”的泛化能力。

---

## 5. 复现

生成评估集：

```bash
python scripts/build_generalization_sets.py \
  --out /tmp/opencode/generalization_fixed \
  --sizes 8,10,12,16,20 \
  --levels L0,L4 \
  --n 100 \
  --budget fixed \
  --same-wall \
  --same-wall-levels L4 \
  --same-wall-n 75
```

长预算集：

```bash
python scripts/build_generalization_sets.py \
  --out /tmp/opencode/generalization_scaled \
  --sizes 16,20 \
  --levels L0,L4 \
  --n 100 \
  --budget scaled
```

评估示例：

```bash
python scripts/sft_agent.py \
  --eval-only \
  --init-adapter /tmp/opencode/grpo_unified_chunk17/grpo_agent.pt \
  --data /tmp/opencode/generalization_fixed/size_20x20 \
  --levels L0,L4 \
  --val-limit 0 \
  --test-limit 100 \
  --eval-batch-size 8 \
  --history-window 12 \
  --max-seq-len 4096 \
  --json /tmp/opencode/gen_eval_size_20.json
```

原始结果：

```text
/tmp/opencode/gen_eval_size_*.json
/tmp/opencode/gen_eval_same_wall.json
```
