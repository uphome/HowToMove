# 训练 / 推理可视化 UI

浏览器页面由 `env/ui.py`（HTTP + SSE + 数据模型）和 `env/ui_page.py`
（纯 HTML/CSS/JS，无外部依赖）组成。默认地址：

```
http://127.0.0.1:7861/
```

---

## 1. 启动方式

### 独立页面 + 训练进程连接

```bash
conda activate minimind
python -m env.ui --port 7861

# 另一个终端：训练把指标和轨迹推到这个 UI
python scripts/train_grpo.py --updates 20 --maps-per-update 1 --g 8 \
    --ui-url http://127.0.0.1:7861
```

### 训练脚本自带 UI

```bash
python scripts/train_grpo.py --updates 20 --maps-per-update 1 --g 8
# 终端会打印 UI: http://127.0.0.1:7862/

# 远程机器（需要放行/转发端口）：
python scripts/train_grpo.py --updates 20 --maps-per-update 1 --g 8 \
    --ui-host 0.0.0.0 --ui-port 7861
```

### 真模型评估 / 回放

```bash
python scripts/eval_llm.py --split test_id --limit 10 \
    --ui-url http://127.0.0.1:7861
```

`scripts/live_rollout.py --serve` 也可以起一个独立 UI。

> `scripts/train_phase1.py` 目前是单步 GRPO 实验，不推送 UI；
> 完整训练看板用 `scripts/train_grpo.py`。

---

## 2. 页面内容

### 训练看板

- 最新指标：update / progress、episodes、success、mean reward、mean steps、
  collisions、format errors、degenerate、samples、loss、pg、kl、ratio、
  clip、drop、grad norm；
- 6 张曲线：loss/pg、mean reward、success/episodes、collisions/format errors、
  ratio min/max、grad norm；
- 最近 30 次更新的表格，含完整数值。

数据来自 `TrainViz.update(**rec_metrics)`，每个 update 一条；同一 update
重复推送会覆盖而不是追加。

### 推理轨迹

- 左侧迷宫实时显示 agent、最短路、当前位置和本步奖励；
- 轨迹下拉框可切换最近 episode；选中后暂停跟随；
- 当前 episode 汇总：level / seed / turn / shortest / detour / success /
  total reward / steps / reason；
- 当前状态：位置、距离、步数、本步奖励、结束原因等；
- 模型输出：原始文本、action、reward、tokens、logprob、latency、finish，
  以及模型实际看到的 prompt；
- 逐步轨迹表：每个 turn 的 action / reward / pos / dist / reason /
  tokens / logp / latency；点击任一行回看该步，用「跟随最新」恢复实时。

---

## 3. HTTP 接口

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 页面 |
| GET | `/api/state` | 当前帧 + 最新训练指标 |
| GET | `/api/stream` | SSE：推理帧 / 训练指标实时推送 |
| GET | `/api/train` | `{"latest": ..., "history": [...]}` |
| GET | `/api/episodes` | 最近 episode（含 `turns`） |
| GET | `/api/episode?id=` | 某 episode 的最新帧 |
| GET | `/api/trace?id=` | 某 episode 的逐 turn 帧 |
| POST | `/api/maze` | `{"level","seed","max_steps"}` 新建地图 |
| POST | `/api/action` | `{"action"}` 手动走一步 |
| POST | `/api/frame` | 外部推入任意帧（训练/评估进程用） |

服务端保留上限：

- 训练历史：500 条 update；
- episode：64 条；
- 每条 episode：64 帧（30 步训练 + done 足够）。

这些都在内存里，UI 服务重启后清空；训练指标/轨迹的原始数据仍会由
`train_grpo.py` 写到 `--json` 指定的文件。
