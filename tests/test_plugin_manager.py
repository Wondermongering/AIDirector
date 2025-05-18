from types import SimpleNamespace

import importlib
from plugin_manager import PluginManager

class DummyConfig:
    def __init__(self, plugins=None):
        self._plugins = plugins or []

    def get_plugins(self):
        return self._plugins


def test_load_from_config():
    cfg = DummyConfig(plugins=["tests.sample_plugin"])
    pm = PluginManager(cfg)
    assert any(getattr(p, "sample_hook", None) for p in pm.plugins)
    result = pm.run_hook("sample_hook", 1, kw=2)
    assert result == ["ok"]
    from tests import sample_plugin
    assert sample_plugin.called == {"arg": 1, "kw": 2}


def test_load_from_entry_points(monkeypatch):
    plugin_module = SimpleNamespace(hello=lambda: "hi")

    class EP:
        name = "ep"
        def load(self):
            return plugin_module

    def fake_entry_points(**kwargs):
        if kwargs.get("group") == "aidirector.plugins":
            return [EP()]
        return []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    pm = PluginManager(DummyConfig())
    assert pm.run_hook("hello") == ["hi"]
