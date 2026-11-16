from __future__ import annotations

from prompts import REL_FEWSHOT, build_messages


def test_default_messages_have_no_system_or_fewshot():
    assert build_messages("state") == [{"role": "user", "content": "state"}]


def test_rel_fewshot_message_structure():
    msgs = build_messages("state", fewshot="rel")
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "state"}
    assert len(msgs) == 1 + 2 * len(REL_FEWSHOT) + 1
    assert [m["role"] for m in msgs] == (
        ["system"] + ["user", "assistant"] * len(REL_FEWSHOT) + ["user"]
    )
    assert "delta=(0,+3)" in msgs[1]["content"]
    assert '"direction": "down"' in msgs[2]["content"]
    assert "delta=(-3,0)" in msgs[-3]["content"]
    assert '"direction": "left"' in msgs[-2]["content"]
