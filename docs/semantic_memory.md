# Semantic Memory

`semantic_memory.py` provides a lightweight placeholder for embedding based memory.
It stores `MemoryItem` objects containing text and vector embeddings.

```
from semantic_memory import SemanticMemory
mem = SemanticMemory()
mem.add("hello world", [0.1, 0.2, 0.3])
results = mem.search([0.1, 0.2, 0.3])
```

This can be expanded with a real similarity search backend for long-term recall.
