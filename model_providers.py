# coding: utf-8
"""Provider abstractions used by ResponseGenerator."""
from __future__ import annotations

import asyncio
import os
import requests
from abc import ABC, abstractmethod
from typing import Any

from rich.prompt import Prompt

from conversation_memory import ConversationMemory
from model_registry import ModelConfig


class ModelProviderInterface(ABC):
    """Abstract interface for model providers."""

    @staticmethod
    def build_messages(system_prompt: str, memory: ConversationMemory) -> list[dict[str, Any]]:
        """Return messages formatted for API calls."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(memory.get_formatted_messages())
        return messages

    @abstractmethod
    async def generate_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
    ) -> str:
        """Return a response from the model."""
        raise NotImplementedError


class OpenAIProvider(ModelProviderInterface):
    """Generate responses using the OpenAI API."""

    async def generate_response(
        self, model_config: ModelConfig, conversation_memory: ConversationMemory, system_prompt: str
    ) -> str:
        try:
            messages = self.build_messages(system_prompt, conversation_memory)
            response = await asyncio.to_thread(
                model_config.client.chat.completions.create,
                model=model_config.api_name,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_response_tokens,
                **model_config.custom_parameters,
            )
            return response.choices[0].message.content
        except Exception as e:  # pragma: no cover - API call
            return f"[Error generating response: {e}]"


class AnthropicProvider(ModelProviderInterface):
    """Generate responses using the Anthropic API."""

    async def generate_response(
        self, model_config: ModelConfig, conversation_memory: ConversationMemory, system_prompt: str
    ) -> str:
        try:
            messages = conversation_memory.get_formatted_messages()
            response = await asyncio.to_thread(
                model_config.client.messages.create,
                model=model_config.api_name,
                system=system_prompt,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_response_tokens,
                **model_config.custom_parameters,
            )
            return response.content[0].text
        except Exception as e:  # pragma: no cover - API call
            return f"[Error generating response: {e}]"


class CLIProvider(ModelProviderInterface):
    """Generate responses by forwarding to a local CLI service."""

    def __init__(self) -> None:
        self.world_interface_key = os.getenv("WORLD_INTERFACE_KEY")

    async def generate_response(
        self, model_config: ModelConfig, conversation_memory: ConversationMemory, system_prompt: str
    ) -> str:
        try:
            messages = self.build_messages(system_prompt, conversation_memory)
            response = await asyncio.to_thread(
                lambda: requests.post(
                    "http://localhost:3000/v1/chat/completions",
                    json={"messages": messages},
                    headers={
                        "Authorization": f"Bearer {self.world_interface_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - API call
            return f"[Error generating response: {e}]"


class HumanProvider(ModelProviderInterface):
    """Obtain responses from a human participant."""

    async def generate_response(
        self, model_config: ModelConfig, conversation_memory: ConversationMemory, system_prompt: str
    ) -> str:
        return await asyncio.to_thread(Prompt.ask, "\n[Human Input]")

