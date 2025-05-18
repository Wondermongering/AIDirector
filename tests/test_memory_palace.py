import numpy as np
import memory_palace

class DummyModel:
    def encode(self, text):
        return np.array([len(text)], dtype=float)

def dummy_loader(name=""):
    return DummyModel()


def test_add_and_retrieve(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_palace, "SentenceTransformer", dummy_loader)
    palace = memory_palace.MemoryPalace(threshold=0.1)
    palace.add_memory("hello world", {"id": 1})
    palace.add_memory("goodbye", {"id": 2})

    results = palace.retrieve_related("hello", top_k=1)
    assert results[0]["metadata"]["id"] == 1

    file_path = tmp_path / "palace.json"
    palace.save(file_path)

    monkeypatch.setattr(memory_palace, "SentenceTransformer", dummy_loader)
    palace2 = memory_palace.MemoryPalace(threshold=0.1)
    palace2.load(file_path)
    assert len(palace2.entries) == 2
