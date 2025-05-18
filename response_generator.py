"""Generate responses from various AI providers."""
import asyncio
import os
import logging
from typing import Optional
import requests

try:  # optional dependency during tests
    from openai import OpenAIError, RateLimitError
except Exception:  # pragma: no cover - fallback when openai not installed
    class OpenAIError(Exception):
        """Fallback OpenAI base error."""

    class RateLimitError(OpenAIError):
        """Fallback error for rate limiting."""

from rich.prompt import Prompt

from conversation_memory import ConversationMemory
from model_registry import ModelRegistry, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses from AI models."""

    MAX_RETRIES = 3

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
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_memory.get_formatted_messages())
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    model_config.client.chat.completions.create,
                    model=model_config.api_name,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_response_tokens,
                    **model_config.custom_parameters,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                wait = 2 ** attempt
                logger.warning(
                    "OpenAI rate limit encountered. Retrying in %s seconds...",
                    wait,
                )
                await asyncio.sleep(wait)
            except OpenAIError as e:
                logger.error("OpenAI API error: %s", e)
                return f"[Error generating response: {e}]"
            except Exception as e:  # pragma: no cover - unexpected
                logger.error("Unexpected error generating OpenAI response: %s", e)
                return f"[Error generating response: {e}]"
        logger.error("Failed to generate OpenAI response after %s attempts", self.MAX_RETRIES)
        return "[Error generating response: retry limit exceeded]"

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
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_memory.get_formatted_messages())
        for attempt in range(self.MAX_RETRIES):
            try:
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
            except requests.exceptions.HTTPError as e:
                if getattr(e.response, "status_code", None) == 429:
                    wait = 2 ** attempt
                    logger.warning(
                        "CLI API rate limited. Retrying in %s seconds...", wait
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("HTTP error from CLI API: %s", e)
                return f"[Error generating response: {e}]"
            except requests.exceptions.RequestException as e:
                logger.error("Request to CLI API failed: %s", e)
                return f"[Error generating response: {e}]"
            except Exception as e:  # pragma: no cover - unexpected
                logger.error("Unexpected error generating CLI response: %s", e)
                return f"[Error generating response: {e}]"
        logger.error("Failed to generate CLI response after %s attempts", self.MAX_RETRIES)
        return "[Error generating response: retry limit exceeded]"

    def _generate_human_response(self) -> str:
        return Prompt.ask("\n[Human Input]")
