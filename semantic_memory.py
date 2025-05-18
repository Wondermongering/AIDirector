"""Simple semantic memory using embeddings."""
from dataclasses import dataclass
from typing import List


@dataclass
class MemoryItem:
    text: str
    embedding: List[float]


class SemanticMemory:
    """In-memory store of embeddings for semantic recall."""

    def __init__(self) -> None:
        self.items: List[MemoryItem] = []

    def add(self, text: str, embedding: List[float]) -> None:
        self.items.append(MemoryItem(text=text, embedding=embedding))

    def search(self, embedding: List[float], top_k: int = 5) -> List[MemoryItem]:
        # Placeholder similarity search. Returns first k items.
        return self.items[:top_k]
