from __future__ import annotations
"""Simple semantic memory using sentence-transformers embeddings."""
from typing import Any, Dict, List, Optional
import json

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency may be missing
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore


class SemanticMemory:
    """Store text entries with embeddings for semantic search."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None or np is None:
            raise ImportError("sentence_transformers and numpy are required")
        self.model = SentenceTransformer(model_name)
        self.embeddings: List[np.ndarray] = []
        self.entries: List[Dict[str, Any]] = []

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed and store a text snippet with optional metadata."""
        embedding = self.model.encode(text)
        self.embeddings.append(np.array(embedding))
        self.entries.append({"text": text, "metadata": metadata or {}})

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` entries most similar to ``query``."""
        if not self.embeddings:
            return []
        query_vec = np.array(self.model.encode(query))
        embs = np.vstack(self.embeddings)
        # cosine similarity
        denom = (np.linalg.norm(embs, axis=1) * np.linalg.norm(query_vec) + 1e-10)
        scores = embs @ query_vec / denom
        idx = np.argsort(-scores)[:top_k]
        return [self.entries[i] for i in idx]

    def save(self, path: str) -> None:
        """Persist the memory to a JSON file."""
        data = []
        for emb, entry in zip(self.embeddings, self.entries):
            data.append({
                "embedding": emb.tolist(),
                "text": entry["text"],
                "metadata": entry["metadata"],
            })
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load memory from a JSON file overwriting current state."""
        with open(path, "r") as f:
            data = json.load(f)
        self.embeddings = [np.array(item["embedding"]) for item in data]
        self.entries = [
            {"text": item["text"], "metadata": item["metadata"]} for item in data
        ]
