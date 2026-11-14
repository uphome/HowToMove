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


def build_messages(state_text: str, system: str | None = None) -> list[dict]:
    sys_text = SYSTEM if system is None else system
    messages = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": state_text})
    return messages


def render_prompt(tokenizer, state_text: str, enable_thinking: bool = False,
                  system: str | None = None) -> tuple[str, list[int]]:
    """返回 (prompt 文本, token ids)。"""
    text = tokenizer.apply_chat_template(
        build_messages(state_text, system),
        tokenize=False,
        add_generation_prompt=True,
        tools=TOOLS,
        enable_thinking=enable_thinking,
    )
    return text, tokenizer.encode(text)


__all__ = ["TOOLS", "SYSTEM", "build_messages", "render_prompt"]
