import sys
from types import SimpleNamespace

class DummyEncoder:
    def encode(self, text):
        # simplistic tokenization by splitting on whitespace
        return text.split()

def _dummy_client(*args, **kwargs):
    return SimpleNamespace()

sys.modules.setdefault("openai", SimpleNamespace(OpenAI=_dummy_client))
sys.modules.setdefault("anthropic", SimpleNamespace(Client=_dummy_client))
sys.modules.setdefault("tiktoken", SimpleNamespace(get_encoding=lambda name='cl100k_base': DummyEncoder()))
