# Agent Runtime：静态地图 + 紧凑 tool result

> 日期：2026-11-17 ｜ 目标：把游戏包装成标准 tool-call agent 对话

---

## 1. 设计

当前 Agent 使用标准消息循环：

```text
system: 静态地图 + 规则 + 工具契约
user: Begin the maze.

assistant: tool_call move(up)
tool: {"moved": "up", "ok": true, "pos": ..., ...}

assistant: tool_call move(left)
tool: {"moved": "left", "ok": false, "pos": ..., ...}
```

起点、终点和墙都只放在 system 中，不再每轮重复完整 observation。

---

## 2. System Prompt

由 `build_agent_system(maze)` 根据地图动态生成：

```text
You are an agent solving a 12x12 maze.
Coordinates: x increases to the right, y increases downward.
Start: (2, 2)
Goal: (10, 6)
Walls: (3,4) (3,5) ...

You control one cell. Each turn call move with one direction.

The move tool returns compact JSON:
moved / ok / pos / blocked / blocked_by / done / goal_delta / reason.

If you hit a wall or the map boundary, your position does not change.
If done=true, stop calling move.
Always call move. Never reply in plain text.
```

后续加入 few-shot 时会单独做对照；第一版不加，避免地图示例混淆。

---

## 3. Tool Result

正常移动：

```json
{
  "moved": "up",
  "ok": true,
  "pos": [4, 8],
  "blocked": false,
  "blocked_by": null,
  "done": false,
  "goal_delta": [6, -2],
  "reason": "You moved up."
}
```

撞墙：

```json
{
  "moved": "up",
  "ok": false,
  "pos": [4, 8],
  "blocked": true,
  "blocked_by": "wall",
  "done": false,
  "goal_delta": [6, -2],
  "reason": "You hit a wall. Your position did not change."
}
```

撞边界：

```json
{
  "moved": "left",
  "ok": false,
  "pos": [0, 8],
  "blocked": true,
  "blocked_by": "boundary",
  "done": false,
  "reason": "You hit the map boundary. Your position did not change."
}
```

到达终点：

```json
{
  "moved": "right",
  "ok": true,
  "pos": [10, 6],
  "blocked": false,
  "blocked_by": null,
  "done": true,
  "goal_delta": [0, 0],
  "reason": "You reached the goal."
}
```

碰撞时位置保持不变，`blocked_by` 明确区分墙和边界。

---

## 4. History

每条轨迹的消息结构：

```text
system
user: Begin the maze.
assistant: tool_call
tool: compact JSON
assistant: tool_call
tool: compact JSON
...
```

默认保留最近 12 对 `assistant/tool`：

```python
history_window = 12
```

更早的消息会从历史头删除，但 system 中的静态地图始终保留。

---

## 5. 推理

`ChatPolicy`：

- 用 `tokenizer.apply_chat_template(..., tools=TOOLS)`；
- 使用自定义 `RolloutEngine`；
- 从 tool call 解析 direction；
- `enable_thinking=False`，第一版不启用思考链。

后续可做 thinking on/off A/B，但不和 tool-result 格式实验同时进行。

---

## 6. 文件

```text
agent/policy.py
agent/runtime.py
scripts/agent_play.py
test_agent.py
```

运行：

```bash
python scripts/agent_play.py \
  --level L0 --seed 7 \
  --history-window 12 \
  --max-seq-len 4096 \
  --json /tmp/opencode/agent_static_l0.json
```

---

## 7. 测试

`test_agent.py` 覆盖：

- system 包含 start/goal/walls；
- user 初始消息是 `Begin the maze.`；
- assistant/tool 历史追加；
- tool result 的 pos/blocked/reason；
- 撞墙位置不变；
- goal reason；
- history window 裁剪。

全量：

```text
92 passed
```

---

## 8. Base 模型实测

```text
level = L0
seed = 7
steps = 12
success = False
format_errors = 1
collisions = 0
```

base 模型在静态地图 + 紧凑 tool result 下仍然会循环：

```text
up / down / left / right / up / down ...
```

这不是 agent 协议的问题，而是 base 模型还没有用这种格式训练过。
tool result 已经正确进入上下文，例如：

```json
{
  "moved": "up",
  "ok": true,
  "pos": [2, 1],
  "blocked": false,
  "blocked_by": null,
  "done": false,
  "goal_delta": [8, 5],
  "reason": "You moved up."
}
```

下一步是使用同一套 system/tool-result 格式生成 SFT 数据。

---

## 9. 下一步

1. 用同一格式生成 SFT 数据；
2. 只对 assistant tool_call token 算 loss；
3. 先 L0，再 L1；
4. 评估模型是否真的会读取 `pos/blocked/goal_delta`；
5. 最后再试 GRPO。
