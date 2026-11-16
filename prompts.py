"""LLM 侧的 prompt 组装（观测文本 → Qwen3 chat template + tools）。

唯一要求：**训练和推理必须用同一个函数**。任何一处用了 `format_chat`
手工拼接、另一处走官方 Jinja 模板，都会让模型看到的输入悄悄变形。
"""

from __future__ import annotations

TOOLS = [{
    "type": "function",
    "function": {
        "name": "move",
        "description": ("Always call this tool to move the agent one cell in the "
                        "given direction. Never reply in plain text."),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                },
            },
            "required": ["direction"],
        },
    },
}]

# 默认不注入自定义 system：工具说明由 Qwen3 模板自动生成，
# 观测本身就是全部上下文（出口 + 障碍坐标 + 当前位置）。
SYSTEM = ""

REL_SYSTEM = (
    "You are solving a 12x12 grid maze. Coordinates are relative to your "
    "current cell: origin=you, x increases right, y increases down. "
    "goal: delta=(dx,dy) is the exit relative to you. "
    "bounds: left/right/up/down are steps to the four edges. "
    "walls_rel lists walls relative to you. "
    "Always call the move tool with exactly one direction."
)


def _tool_call_text(direction: str) -> str:
    return ('<tool_call>\n'
            '{"name": "move", "arguments": {"direction": "%s"}}\n'
            '</tool_call>' % direction)


# 四个 cardinal 例子，明确教 dx/dy 的符号与动作映射。
REL_FEWSHOT = [
    ("map 12x12, x=right y=down, origin=you\n"
     "goal: delta=(0,+3)\n"
     "bounds: left=3 right=8 up=4 down=7\n"
     "walls_rel: none", "down"),
    ("map 12x12, x=right y=down, origin=you\n"
     "goal: delta=(0,-3)\n"
     "bounds: left=3 right=8 up=4 down=7\n"
     "walls_rel: none", "up"),
    ("map 12x12, x=right y=down, origin=you\n"
     "goal: delta=(+3,0)\n"
     "bounds: left=3 right=8 up=4 down=7\n"
     "walls_rel: none", "right"),
    ("map 12x12, x=right y=down, origin=you\n"
     "goal: delta=(-3,0)\n"
     "bounds: left=3 right=8 up=4 down=7\n"
     "walls_rel: none", "left"),
]

# L1 额外给两条带墙示例（来自 L1 balanced train 的真实状态）。
REL_FEWSHOT_L1 = REL_FEWSHOT + [
    ("map 12x12, x=right y=down, origin=you\n"
     "goal: delta=(+0,+6)\n"
     "bounds: left=1 right=10 up=3 down=8\n"
     "walls_rel: (+1,+2) (+1,+4) (+1,+5) (+1,+6) (+1,+7) (+1,+8)", "down"),
    ("map 12x12, x=right y=down, origin=you\n"
     "goal: delta=(-7,+1)\n"
     "bounds: left=9 right=2 up=3 down=8\n"
     "walls_rel: (-2,-1) (-2,+1) (-2,-2) (-2,-3)", "left"),
]


def build_messages(state_text: str, system: str | None = None,
                   fewshot: str | None = None) -> list[dict]:
    if fewshot in ("rel", "rel_l1"):
        sys_text = REL_SYSTEM
    else:
        sys_text = SYSTEM if system is None else system
    messages = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    if fewshot in ("rel", "rel_l1"):
        examples = REL_FEWSHOT_L1 if fewshot == "rel_l1" else REL_FEWSHOT
        for example_state, direction in examples:
            messages.append({"role": "user", "content": example_state})
            messages.append({"role": "assistant",
                             "content": _tool_call_text(direction)})
    messages.append({"role": "user", "content": state_text})
    return messages


def render_prompt(tokenizer, state_text: str, enable_thinking: bool = False,
                  system: str | None = None,
                  fewshot: str | None = None) -> tuple[str, list[int]]:
    """返回 (prompt 文本, token ids)。"""
    text = tokenizer.apply_chat_template(
        build_messages(state_text, system, fewshot=fewshot),
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        enable_thinking=enable_thinking,
    )
    return text, tokenizer.encode(text)


__all__ = [
    "TOOLS", "SYSTEM", "REL_SYSTEM", "REL_FEWSHOT", "REL_FEWSHOT_L1",
    "build_messages", "render_prompt",
]
