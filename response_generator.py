"""Generate responses from various AI providers."""
import asyncio
import os
import requests
import logging
from typing import Optional

from rich.prompt import Prompt

from conversation_memory import ConversationMemory
from model_registry import ModelRegistry, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses from AI models."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.world_interface_key = os.getenv("WORLD_INTERFACE_KEY")

    async def generate_response(
        self,
        model_name: str,
        conversation_memory: ConversationMemory,
        system_prompt: Optional[str] = None,
    ) -> str:
        model_config = self.registry.get_model(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        effective_prompt = system_prompt or model_config.system_prompt

        if model_config.provider == ModelProvider.OPENAI:
            return await self._generate_openai_response(
                model_config, conversation_memory, effective_prompt
            )
        if model_config.provider == ModelProvider.ANTHROPIC:
            return await self._generate_anthropic_response(
                model_config, conversation_memory, effective_prompt
            )
        if model_config.provider == ModelProvider.CLI:
            return await self._generate_cli_response(
                model_config, conversation_memory, effective_prompt
            )
        if model_config.provider == ModelProvider.HUMAN:
            return self._generate_human_response()
        raise ValueError(f"Unsupported provider: {model_config.provider}")

    async def _generate_openai_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
    ) -> str:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_memory.get_formatted_messages())
            response = await asyncio.to_thread(
                model_config.client.chat.completions.create,
                model=model_config.api_name,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_response_tokens,
                **model_config.custom_parameters,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Error generating OpenAI response: %s", e)
            return f"[Error generating response: {e}]"

    async def _generate_anthropic_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
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
        except Exception as e:
            logger.error("Error generating Anthropic response: %s", e)
            return f"[Error generating response: {e}]"

    async def _generate_cli_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
    ) -> str:
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_memory.get_formatted_messages())
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
        except Exception as e:
            logger.error("Error generating CLI response: %s", e)
            return f"[Error generating response: {e}]"

    def _generate_human_response(self) -> str:
        return Prompt.ask("\n[Human Input]")
