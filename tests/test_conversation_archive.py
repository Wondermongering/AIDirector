import numpy as np
import semantic_memory
import conversation_archive
from conversation_memory import Message

class DummyModel:
    def encode(self, text):
        return np.array([len(text)], dtype=float)

def dummy_loader(name=""):
    return DummyModel()

def create_message(text):
    return Message(role="user", content=text)

def test_add_and_search(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic_memory, "SentenceTransformer", dummy_loader)
    archive_path = tmp_path / "archive.json"
    archive = conversation_archive.ConversationArchive(archive_path)
    archive.add_conversation([create_message("hello world"), create_message("foo")], {"id": 1})
    archive.add_conversation([create_message("goodbye")], {"id": 2})

    results = archive.search("hello", top_k=1)
    assert results[0]["metadata"]["id"] == 1

    archive.save()
    monkeypatch.setattr(semantic_memory, "SentenceTransformer", dummy_loader)
    archive2 = conversation_archive.ConversationArchive(archive_path)
    assert len(archive2.memory.entries) == 2
