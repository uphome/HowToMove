from __future__ import annotations

from prompts import REL_FEWSHOT, REL_FEWSHOT_L1, build_messages


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
    user_text = "\n".join(m["content"] for m in msgs if m["role"] == "user")
    assert "delta=(-3,0)" in user_text
    assert "delta=(+3,-2)" in user_text
    assert '"direction": "left"' in "\n".join(
        m["content"] for m in msgs if m["role"] == "assistant")


def test_rel_l1_fewshot_adds_wall_examples():
    msgs = build_messages("state", fewshot="rel_l1")
    assert len(msgs) == 1 + 2 * len(REL_FEWSHOT_L1) + 1
    assert len(REL_FEWSHOT_L1) == len(REL_FEWSHOT) + 2
    assert any("walls_rel: (+1,+2)" in m["content"] for m in msgs if m["role"] == "user")
    assert any("walls_rel: (+1,-1) (+2,-1) (+3,-1)" in m["content"]
               for m in msgs if m["role"] == "user")
