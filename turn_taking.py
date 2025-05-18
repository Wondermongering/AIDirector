"""Dialogic turn-taking coordinator."""

from __future__ import annotations

import asyncio
from typing import List

from conversation_memory import ConversationMemory, Message, MessageContent
from model_registry import ModelRegistry, ModelProvider
from response_generator import ResponseGenerator
from memory_palace import MemoryPalace


class DialogicTurnCoordinator:
    """Coordinate a dialogic turn among multiple models."""

    def __init__(
        self,
        models: List[str],
        registry: ModelRegistry,
        generator: ResponseGenerator,
        memory_palace: MemoryPalace,
        max_cycles: int = 3,
    ) -> None:
        """Create a coordinator for ``models`` using ``memory_palace`` for context."""
        self.models = models
        self.registry = registry
        self.generator = generator
        self.memory_palace = memory_palace
        self.max_cycles = max_cycles

    async def execute(self, shared_memory: ConversationMemory) -> None:
        """Run a dialogic turn until interrupt or max cycles."""
        cycle = 0
        idx = 0
        while cycle < self.max_cycles:
            if not self.models:
                break
            model_name = self.models[idx]
            model_config = self.registry.get_model(model_name)
            if not model_config:
                idx = (idx + 1) % len(self.models)
                cycle += 1
                continue

            last_msg = shared_memory.get_messages()[-1]
            if isinstance(last_msg.content, MessageContent):
                query = last_msg.content.text
            else:
                query = last_msg.content
            ctx_items = self.memory_palace.retrieve_related(query, top_k=3)
            ctx_text = "\n".join(item["text"] for item in ctx_items)
            prompt = model_config.system_prompt
            if ctx_text:
                prompt = f"{prompt}\nRelevant context:\n{ctx_text}"

            response_text = await self.generator.generate_response(
                model_name, shared_memory, prompt
            )
            message = Message(
                role="assistant"
                if model_config.provider != ModelProvider.HUMAN
                else "user",
                content=response_text,
                metadata={"model": model_name, "role": model_config.role.value},
            )
            shared_memory.add_message(message)
            text = (
                message.content.text
                if isinstance(message.content, MessageContent)
                else message.content
            )
            self.memory_palace.add_memory(text, {"actor": model_name})

            if "[INTERRUPT]" in response_text.upper():
                break

            idx = (idx + 1) % len(self.models)
            cycle += 1

