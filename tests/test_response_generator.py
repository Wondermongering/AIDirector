import types
import asyncio

import requests
import pytest

from conversation_memory import ConversationMemory
from model_registry import ModelConfig, ModelProvider
from response_generator import ResponseGenerator, OpenAIError, RateLimitError


class DummyRegistry:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_model(self, name):
        return self.cfg


@pytest.mark.asyncio
async def test_openai_rate_limit_retry(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    class DummyRateLimit(RateLimitError):
        pass

    class DummyOpenAIError(OpenAIError):
        pass

    dummy_client = types.SimpleNamespace()
    attempts = {"n": 0}

    def create(**_):
        if attempts["n"] == 0:
            attempts["n"] += 1
            raise DummyRateLimit("rate limit")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )

    dummy_client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    cfg = ModelConfig(api_name="dummy", client=dummy_client, provider=ModelProvider.OPENAI)
    gen = ResponseGenerator(DummyRegistry(cfg))
    result = await gen.generate_response("dummy", ConversationMemory(), "sys")
    assert result == "ok"


@pytest.mark.asyncio
async def test_cli_http_retry(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    attempts = {"n": 0}

    def fake_post(*_, **__):
        attempts["n"] += 1

        class Resp:
            def __init__(self, status):
                self.status_code = status

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.exceptions.HTTPError(response=self)

            def json(self):
                return {"choices": [{"message": {"content": "cli"}}]}

        return Resp(429 if attempts["n"] == 1 else 200)

    monkeypatch.setattr(requests, "post", fake_post)

    cfg = ModelConfig(api_name="dummy", client=None, provider=ModelProvider.CLI)
    gen = ResponseGenerator(DummyRegistry(cfg))
    result = await gen.generate_response("dummy", ConversationMemory(), "sys")
    assert result == "cli"
