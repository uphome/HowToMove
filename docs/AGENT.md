# Agent Runtime 实现说明

> 代码：`agent/`、`scripts/agent_play.py`、`scripts/agent_group_play.py`
> 目标：把 12×12 迷宫包装成标准 tool-call agent，支持单轨迹和多轨迹并行。

---

## 1. 为什么要单独做 Agent

早先的 rollout 是：

```text
state prompt → move → 下一个 state prompt
```

模型每一轮只看到当前状态，看不到自己真实的：

```text
assistant tool_call → tool result
```

因此它不知道完整路径，也不会读取工具返回。

现在改成标准消息循环：

```text
system
user: Begin the maze.

assistant: <tool_call>move(up)</tool_call>
tool: {"moved":"up","ok":true,"pos":[4,8],...}

assistant: <tool_call>move(left)</tool_call>
tool: {"moved":"left","ok":false,"pos":[4,8],...}
```

地图只放在 system 中；后续只返回紧凑的 tool result。

---

## 2. 文件结构

```text
agent/
├── __init__.py
├── policy.py      ChatPolicy：messages → prompt → engine → Step
├── runtime.py     MazeAgent：单条轨迹状态与 apply
└── group.py       AgentGroup：G 条轨迹批量协调

scripts/
├── agent_play.py        单条轨迹 CLI
└── agent_group_play.py  同图 G 条轨迹 CLI

test_agent.py
```

---

## 3. System Prompt：静态地图

`agent/runtime.py` 的 `build_agent_system(maze)` 根据地图动态生成：

```text
You are an agent solving a 12x12 maze.
Coordinates: x increases to the right, y increases downward.
Start: (2, 2)
Goal: (10, 6)
Walls: (3,4) (3,5) ...

You control one cell. Each turn, call the move tool with exactly one
direction: up, down, left, or right.

The move tool returns a compact JSON result:
- moved: the direction you attempted
- ok: whether the move succeeded
- pos: your position after the move
- blocked: whether the move was blocked
- blocked_by: wall, boundary, or null
- done: whether the episode ended
- goal_delta: goal - pos
- reason: a short explanation

If you hit a wall or the map boundary, your position does not change.
If done=true, stop calling move.
Always call move. Never reply in plain text.
```

地图不同，system prompt 就不同。

初始 user 只有：

```text
Begin the maze.
```

不再每轮重复 walls、goal、bounds。

---

## 4. Tool Result

工具名只有：

```text
move(direction: up | down | left | right)
```

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

关键设计：

- 不返回完整 `observation`；
- 不重复 walls/bounds；
- 碰撞时 `pos` 不变；
- `blocked_by` 区分 wall / boundary；
- `reason` 给模型明确反馈。

---

## 5. MazeAgent

`agent/runtime.py`。

### 初始化

```python
agent = MazeAgent(
    maze,
    policy,                 # 可为 None，AgentGroup 中由协调器调用
    history_window=12,
    memory=3,
    revisit_penalty=0.0,
    episode_id="u1/m0/g0",
    on_step=callback,
)
```

内部状态：

```text
env
path
history
turns
total_reward
base_messages
```

`base_messages` 只有：

```text
system
user: Begin the maze.
```

### 单步 apply

```python
step = policy.act(agent.messages())
agent.apply(step)
```

`apply()` 做：

1. 保存当前 prompt；
2. 追加 assistant tool_call；
3. 解析 direction；
4. 计算 blocked_by（wall / boundary）；
5. `env.step(action)`；
6. 更新 path / total_reward；
7. 生成 compact tool result；
8. 追加 tool 消息；
9. 裁剪 history；
10. 记录 turn。

`turn` 里保存：

```text
prompt
prompt_ids
completion_tokens
token_logprobs
action
reward
done
info
tool_result
assistant_text
logprob
```

这些字段供后续 SFT/GRPO 使用。

### 单轨迹 run

```python
run = agent.run()
```

等价于：

```python
while not agent.done:
    step = policy.act(agent.messages())
    agent.apply(step)
```

---

## 6. ChatPolicy

`agent/policy.py`。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=TOOLS,
    enable_thinking=False,
)
ids = tokenizer.encode(text)
samples = engine.generate_batch([ids], sampling)
```

批量版本：

```python
steps = policy.act_batch(messages_batch)
```

内部：

```text
messages_batch
→ 每个 messages 套 chat template
→ 每个 prompt tokenize
→ engine.generate_batch 一次
→ 解析每条 tool_call
→ Step 列表
```

注意：

- 使用自定义 `RolloutEngine`；
- 不是 HuggingFace `generate`；
- `tool` 消息由 Qwen3 模板渲染成 `<tool_response>`；
- `enable_thinking=False`，第一版关闭思考链。

---

## 7. AgentGroup：同图 G 条轨迹

`agent/group.py`。

```python
group = AgentGroup(
    maze,
    policy,
    g=8,
    history_window=12,
    episode_prefix="u1/m0",
)
runs = group.run()
```

核心循环：

```python
active = agents

while active:
    messages_batch = [a.messages() for a in active]
    steps = policy.act_batch(messages_batch)

    for agent, step in zip(active, steps):
        agent.apply(step)

    active = [a for a in active if not a.done]
```

重要：

- 8 个 `MazeAgent` 独立；
- 8 条 history 不共享；
- 每轮只 batch active 轨迹；
- 不用 Python 线程；
- 一个 `RolloutEngine`；
- 用 `episode_id` 保留 group 关系。

当某个 agent 提前结束后，batch 自然缩小。

实际 L0/G=2 验证：

```text
g0 t01 up
g1 t01 up

g0 t02 left
g1 t02 right
```

第二轮历史已经不同，但仍然在同一个生成 batch 中。

---

## 8. 单条运行

```bash
python scripts/agent_play.py \
  --level L0 --seed 7 \
  --history-window 12 \
  --max-seq-len 4096 \
  --json /tmp/opencode/agent_l0.json
```

## 9. 同图 G 条并行

```bash
python scripts/agent_group_play.py \
  --level L0 --seed 7 \
  --g 8 \
  --history-window 12 \
  --temperature 1.2 \
  --max-seq-len 4096 \
  --json /tmp/opencode/agent_group_l0.json
```

---

## 10. 测试

`test_agent.py` 覆盖：

- system 包含 start/goal/walls；
- user 是 `Begin the maze.`；
- assistant/tool 历史；
- compact tool result；
- 撞墙位置不变；
- goal reason；
- history window；
- AgentGroup batch 与 active 缩小。

全量：

```text
93 passed
```

---

## 11. 当前限制

- base 模型还没有用新格式训练，仍会循环；
- 目前只有 L0 小规模真实 Agent 验证；
- SFT 数据生成尚未接入；
- GRPO 的 `samples_from_agent_run` 尚未实现；
- thinking 关闭；
- few-shot 暂未加入，后续做对照实验。

---

## 12. 下一步

```text
1. 用同样的 messages 格式生成 SFT 数据
2. 只监督 assistant tool_call token
3. 先 L0 SFT
4. 评估模型是否读取 pos/blocked/goal_delta
5. 再 L1
6. 接入 AgentGroup + GRPO
```
