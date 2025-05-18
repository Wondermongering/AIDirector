"""Dialogic turn-taking coordinator."""

from __future__ import annotations

import asyncio
from typing import List

from conversation_memory import ConversationMemory, Message
from model_registry import ModelRegistry, ModelProvider
from response_generator import ResponseGenerator


class DialogicTurnCoordinator:
    """Coordinate a dialogic turn among multiple models."""

    def __init__(
        self,
        models: List[str],
        registry: ModelRegistry,
        generator: ResponseGenerator,
        max_cycles: int = 3,
    ) -> None:
        self.models = models
        self.registry = registry
        self.generator = generator
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

            prompt = model_config.system_prompt
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

            if "[INTERRUPT]" in response_text.upper():
                break

            idx = (idx + 1) % len(self.models)
            cycle += 1

