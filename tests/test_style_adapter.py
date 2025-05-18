from style_adapter import StyleAdapter
from conversation_memory import Message


def test_apply_style_changes_prompt():
    adapter = StyleAdapter()
    adapter.update(Message(role="user", content="hello there!"))
    adapter.update(Message(role="user", content="please explain this in detail"))

    prompt = adapter.apply_style("You are an assistant.")
    assert "formal" in prompt or "friendly" in prompt
    assert "detailed" in prompt or "concise" in prompt
