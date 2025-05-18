"""Generate responses from various AI providers."""
import asyncio
import os
import requests
import logging
from time import perf_counter
from typing import Optional

from rich.prompt import Prompt

from conversation_memory import ConversationMemory
from model_registry import ModelRegistry, ModelConfig, ModelProvider
from metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses from AI models."""

    def __init__(self, registry: ModelRegistry, metrics: MetricsCollector | None = None) -> None:
        self.registry = registry
        self.metrics = metrics
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

        start = perf_counter()
        try:
            if model_config.provider == ModelProvider.OPENAI:
                response_text = await self._generate_openai_response(
                    model_config, conversation_memory, effective_prompt
                )
            elif model_config.provider == ModelProvider.ANTHROPIC:
                response_text = await self._generate_anthropic_response(
                    model_config, conversation_memory, effective_prompt
                )
            elif model_config.provider == ModelProvider.CLI:
                response_text = await self._generate_cli_response(
                    model_config, conversation_memory, effective_prompt
                )
            elif model_config.provider == ModelProvider.HUMAN:
                response_text = self._generate_human_response()
            else:
                raise ValueError(f"Unsupported provider: {model_config.provider}")
            duration = perf_counter() - start
            tokens = len(conversation_memory.encoder.encode(response_text))
            if self.metrics:
                self.metrics.record_response(model_name, duration, tokens)
            return response_text
        except Exception as e:
            if self.metrics:
                self.metrics.record_error(model_name)
            logger.error("Error generating response for %s: %s", model_name, e)
            return f"[Error generating response: {e}]"

    async def _generate_openai_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
    ) -> str:
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

    async def _generate_anthropic_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
    ) -> str:
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

    async def _generate_cli_response(
        self,
        model_config: ModelConfig,
        conversation_memory: ConversationMemory,
        system_prompt: str,
    ) -> str:
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

    def _generate_human_response(self) -> str:
        return Prompt.ask("\n[Human Input]")
