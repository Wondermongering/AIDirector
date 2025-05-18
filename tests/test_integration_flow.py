import asyncio
import cli
import response_generator
import pytest


@pytest.mark.asyncio
async def test_full_conversation_flow(monkeypatch, tmp_path):
    responses = iter([
        "gpt response", "claude response",
        "gpt response 2", "claude response 2",
    ])

    async def fake_generate(self, model_name, conv_mem, system_prompt=None):
        return next(responses)

    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k2")
    monkeypatch.setattr(response_generator.ResponseGenerator, "generate_response", fake_generate)
    monkeypatch.setattr(cli.Confirm, "ask", lambda *a, **kw: True)
    monkeypatch.setattr(cli.ConversationLogger, "log_message", lambda *a, **kw: None)
    monkeypatch.setattr(cli.console, "print", lambda *a, **kw: None)

    orch = cli.AIOrchestrator(models=["gpt-4", "claude-3"], max_turns=2, log_folder=str(tmp_path))
    await orch.run_conversation()

    # starter message + 4 generated messages
    assert len(orch.shared_memory.get_messages()) == 5
    for name in ["gpt-4", "claude-3"]:
        assert len(orch.conversation_memories[name].get_messages()) == 3

