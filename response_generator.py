"""Generate responses from various providers."""
import logging
from typing import Optional

from conversation_memory import ConversationMemory
from model_registry import ModelRegistry, ModelConfig, ModelProvider
from model_providers import ModelProviderInterface

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses from AI models."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.providers = registry.providers

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

        provider = self.providers.get(model_config.provider)
        if not provider:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        return await provider.generate_response(
            model_config, conversation_memory, effective_prompt
        )
