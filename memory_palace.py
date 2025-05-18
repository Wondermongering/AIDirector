from __future__ import annotations
"""Organize semantic memories into thematic rooms."""
from typing import Any, Dict, List, Optional
import json

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency may be missing
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore


class MemoryPalace:
    """Cluster memories by semantic similarity."""

    def __init__(self, threshold: float = 0.6, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None or np is None:
            raise ImportError("sentence_transformers and numpy are required")
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.rooms: List[List[int]] = []
        self.room_centroids: List[np.ndarray] = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        return float(a @ b / denom)

    def _update_centroid(self, room_index: int) -> None:
        ids = self.rooms[room_index]
        self.room_centroids[room_index] = np.mean([self.embeddings[i] for i in ids], axis=0)

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Embed ``text`` and assign it to a room."""
        embedding = np.array(self.model.encode(text))
        idx = len(self.entries)
        self.entries.append({"text": text, "metadata": metadata or {}})
        self.embeddings.append(embedding)

        if not self.rooms:
            self.rooms.append([idx])
            self.room_centroids.append(embedding)
            return

        sims = [self._cosine_similarity(embedding, c) for c in self.room_centroids]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.threshold:
            self.rooms[best_idx].append(idx)
            self._update_centroid(best_idx)
        else:
            self.rooms.append([idx])
            self.room_centroids.append(embedding)

    def retrieve_related(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` memories closest to ``query``."""
        if not self.entries:
            return []
        query_vec = np.array(self.model.encode(query))
        sims = [self._cosine_similarity(query_vec, c) for c in self.room_centroids]
        room_order = np.argsort(sims)[::-1]
        results = []
        for r in room_order:
            for idx in self.rooms[r]:
                results.append(self.entries[idx])
                if len(results) >= top_k:
                    return results
        return results

    def save(self, path: str) -> None:
        """Persist palace state to ``path``."""
        data = {
            "entries": self.entries,
            "embeddings": [emb.tolist() for emb in self.embeddings],
            "rooms": self.rooms,
            "room_centroids": [cent.tolist() for cent in self.room_centroids],
            "threshold": self.threshold,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load palace state from ``path`` overwriting current state."""
        with open(path, "r") as f:
            data = json.load(f)
        self.entries = data.get("entries", [])
        self.embeddings = [np.array(e) for e in data.get("embeddings", [])]
        self.rooms = data.get("rooms", [])
        self.room_centroids = [np.array(c) for c in data.get("room_centroids", [])]
        self.threshold = data.get("threshold", self.threshold)
