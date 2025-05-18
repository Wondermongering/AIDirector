from __future__ import annotations
"""Archive conversations with semantic search."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from conversation_memory import Message, MessageContent
from semantic_memory import SemanticMemory


class ConversationArchive:
    """Stores entire conversations and enables semantic search."""

    def __init__(self, path: str | Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.path = Path(path)
        self.memory = SemanticMemory(model_name)
        if self.path.exists():
            self.load(self.path)

    def add_conversation(
        self, messages: List[Message], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a conversation transcript to the archive."""
        transcript_parts = []
        for msg in messages:
            if isinstance(msg.content, MessageContent):
                transcript_parts.append(msg.content.text)
            else:
                transcript_parts.append(str(msg.content))
        transcript = "\n".join(transcript_parts)
        self.memory.add_memory(transcript, metadata or {})

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return conversations most relevant to ``query``."""
        return self.memory.retrieve_relevant(query, top_k=top_k)

    def save(self, path: str | Path | None = None) -> None:
        """Persist the archive to disk."""
        self.memory.save(str(path or self.path))

    def load(self, path: str | Path | None = None) -> None:
        """Load the archive from disk."""
        self.memory.load(str(path or self.path))
