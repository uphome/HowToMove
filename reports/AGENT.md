# Agent Runtime：显式 tool call / tool result

> 日期：2026-11-17 ｜ 目标：把“环境多步、模型单步”改成标准 agent 对话循环

---

## 1. 目标

之前的 rollout 是：

```text
state prompt → move → 下一个 state prompt
```

模型看不到上一轮真实的 assistant tool_call 和 tool result。

现在改成：

```text
system
user: initial observation
assistant: tool_call move(...)
tool: {"moved": ..., "reward": ..., "observation": ...}
assistant: tool_call move(...)
tool: ...
```

---

## 2. 新增文件

```text
agent/__init__.py
agent/policy.py
agent/runtime.py
scripts/agent_play.py
test_agent.py
```

### agent/policy.py

`ChatPolicy`：

- 接收完整 messages；
- 使用 `tokenizer.apply_chat_template(..., tools=TOOLS)`；
- 调用自定义 `RolloutEngine`；
- 解析 `move` tool_call。

### agent/runtime.py

`MazeAgent`：

- 维护 `system + fewshot + initial user + history`；
- 每轮追加 assistant tool_call；
- 执行动作后追加 `tool` 消息；
- 支持 `history_window`，保留最近 K 对 assistant/tool；
- 返回 `AgentRun`，包含 messages、turns、success、reward。

### scripts/agent_play.py

单局 CLI：

```bash
python scripts/agent_play.py \
  --level L0 --seed 7 \
  --obs-style rel --fewshot rel \
  --history-window 12 \
  --max-seq-len 4096
```

---

## 3. Message 格式

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "map 12x12 ..."},
  {
    "role": "assistant",
    "content": "<tool_call>\n{\"name\":\"move\",\"arguments\":{\"direction\":\"up\"}}\n</tool_call>"
  },
  {
    "role": "tool",
    "name": "move",
    "content": "{\"ok\":true,\"moved\":\"up\",\"pos\":[4,8],\"reward\":0.9,\"done\":false,\"observation\":\"...\"}"
  }
]
```

Qwen3 chat template 会把 `tool` 消息渲染成：

```text
<tool_response>...</tool_response>
```

这一点已经单独验证通过。

---

## 4. Tool result

`move` 工具返回：

```json
{
  "ok": true,
  "moved": "up",
  "from": [4, 9],
  "pos": [4, 8],
  "reward": 0.9,
  "done": false,
  "reason": "move",
  "observation": "map 12x12 ...\nlast=up\nvisited_rel=..."
}
```

`observation` 是下一轮完整的状态文本，和当前 `rel` 观测一致。

---

## 5. 历史窗口

默认：

```text
history_window = 12
```

保留最近 12 对 assistant/tool 消息；更早的 tool result 会被丢弃，但 `visited_rel` 仍然保留最近足迹。

第一版没有直接把 30 步全部塞进 prompt，先控制 KV 和训练成本。

---

## 6. L0 base 模型验证

命令：

```bash
python scripts/agent_play.py \
  --level L0 --seed 7 \
  --obs-style rel --fewshot rel \
  --history-window 12 \
  --max-seq-len 4096 \
  --json /tmp/opencode/agent_l0_seed7.json
```

结果：

```text
steps = 30
success = False
format_errors = 0
collisions = 2
```

轨迹仍然是：

```text
right right ... right collision
left  left  ... left
right ...
```

这说明：

- agent 格式和 tool result 渲染正常；
- base 模型还没有用新格式 SFT，看到 tool history 也不会自动学会使用；
- 下一步必须用同样的 message 格式重新生成 SFT 数据。

---

## 7. 测试

新增：

```text
test_agent.py
```

覆盖：

- assistant/tool 历史是否正确追加；
- tool result 是否包含 `moved/pos/reward/observation`；
- history_window 是否只保留最近对；
- 两次动作后是否成功到达 goal。

全量测试：

```text
90 passed
```

---

## 8. 下一步

这一步只实现 runtime，不改训练。

下一阶段：

1. 用同一个 `MazeAgent` 消息格式生成 SFT 对话数据；
2. 只对 assistant tool_call token 算 loss；
3. 重新跑 L0 SFT；
4. 验证模型是否真的会读取 tool result；
5. 再推进 L1 和新格式 GRPO。
