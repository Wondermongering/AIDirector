import asyncio
from types import SimpleNamespace

import pytest
import requests

import conversation_memory
import model_registry
import response_generator


@pytest.mark.asyncio
async def test_openai_error(monkeypatch):
    def fail_create(*a, **k):
        raise RuntimeError("fail")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail_create)))
    cfg = model_registry.ModelConfig(
        api_name="gpt-4",
        client=client,
        provider=model_registry.ModelProvider.OPENAI,
    )
    rg = response_generator.ResponseGenerator(SimpleNamespace(get_model=lambda n: cfg))
    mem = conversation_memory.ConversationMemory()
    result = await rg._generate_openai_response(cfg, mem, "sys")
    assert result.startswith("[Error generating response:")


@pytest.mark.asyncio
async def test_cli_error(monkeypatch):
    def fail_post(*a, **k):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "post", fail_post)

    cfg = model_registry.ModelConfig(
        api_name="test-cli",
        client=None,
        provider=model_registry.ModelProvider.CLI,
    )
    rg = response_generator.ResponseGenerator(SimpleNamespace(get_model=lambda n: cfg))
    mem = conversation_memory.ConversationMemory()
    result = await rg._generate_cli_response(cfg, mem, "sys")
    assert result.startswith("[Error generating response:")


@pytest.mark.asyncio
async def test_unknown_model(monkeypatch):
    rg = response_generator.ResponseGenerator(SimpleNamespace(get_model=lambda n: None))
    mem = conversation_memory.ConversationMemory()
    with pytest.raises(ValueError):
        await rg.generate_response("missing", mem)

