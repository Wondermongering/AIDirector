import main

class DummyConfig:
    def get_model_configs(self):
        return {}


def test_registry_with_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key2")
    registry = main.ModelRegistry(DummyConfig())
    models = registry.list_available_models()
    assert "gpt-4" in models
    assert "claude-3" in models
    assert "human" in models


def test_registry_without_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    registry = main.ModelRegistry(DummyConfig())
    models = registry.list_available_models()
    assert "gpt-4" not in models
    assert "claude-3" not in models
    assert "human" in models

