import semantic_memory
import numpy as np

class DummyModel:
    def encode(self, text):
        # simple "embedding" based on length
        return np.array([len(text)], dtype=float)

def dummy_loader(name=""):
    return DummyModel()


def test_add_and_retrieve(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic_memory, "SentenceTransformer", dummy_loader)
    mem = semantic_memory.SemanticMemory()
    mem.add_memory("hello world", {"id": 1})
    mem.add_memory("goodbye", {"id": 2})

    results = mem.retrieve_relevant("hello", top_k=1)
    assert results[0]["metadata"]["id"] == 1

    file_path = tmp_path / "mem.json"
    mem.save(file_path)

    monkeypatch.setattr(semantic_memory, "SentenceTransformer", dummy_loader)
    mem2 = semantic_memory.SemanticMemory()
    mem2.load(file_path)
    assert len(mem2.entries) == 2
