"""Model configuration and registry utilities."""
import logging
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import anthropic
from dotenv import load_dotenv
import yaml

from conversation_memory import TokenLimitStrategy
from model_providers import (
    AnthropicProvider,
    CLIProvider,
    HumanProvider,
    ModelProviderInterface,
    OpenAIProvider,
)

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    OPENAI = auto()
    ANTHROPIC = auto()
    CLI = auto()
    HUMAN = auto()

    def __str__(self) -> str:
        return self.name.capitalize()


class ModelRole(Enum):
    PRIMARY = "primary"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    MEDIATOR = "mediator"
    CREATIVE = "creative"
    LOGICAL = "logical"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value.capitalize()


@dataclass
class ModelConfig:
    """Configuration for an AI model."""

    api_name: str
    client: Any
    provider: ModelProvider
    role: ModelRole = ModelRole.PRIMARY
    temperature: float = 0.7
    max_response_tokens: int = 1024
    max_context_tokens: int = 8192
    token_limit_strategy: TokenLimitStrategy = TokenLimitStrategy.TRUNCATE_OLDEST
    system_prompt: str = ""
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role == ModelRole.CUSTOM and not self.system_prompt:
            raise ValueError("Custom role requires a system prompt")


class ConfigurationManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(
                "Configuration file %s not found. Using defaults.", self.config_path
            )
            return {}
        with open(self.config_path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                logger.error("Error parsing configuration file: %s", e)
                return {}

    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get("models", {})

    def get_orchestration_config(self) -> Dict[str, Any]:
        return self.config.get("orchestration", {})

    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get("logging", {})


class ModelRegistry:
    """Registry for AI models with initialization of clients."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        load_dotenv()
        self.config_manager = config_manager
        self.models: Dict[str, ModelConfig] = {}
        self.providers: dict[ModelProvider, ModelProviderInterface] = {
            ModelProvider.OPENAI: OpenAIProvider(),
            ModelProvider.ANTHROPIC: AnthropicProvider(),
            ModelProvider.CLI: CLIProvider(),
            ModelProvider.HUMAN: HumanProvider(),
        }
        self._initialize_models()

    def _initialize_models(self) -> None:
        model_configs = self.config_manager.get_model_configs()

        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            openai_client = None
            logger.warning(
                "OpenAI API key not found. OpenAI models will not be available."
            )

        # Initialize Anthropic client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            anthropic_client = anthropic.Client(api_key=anthropic_api_key)
        else:
            anthropic_client = None
            logger.warning(
                "Anthropic API key not found. Anthropic models will not be available."
            )

        default_models = {
            "gpt-4": ModelConfig(
                api_name="gpt-4",
                client=openai_client,
                provider=ModelProvider.OPENAI,
                max_context_tokens=8192,
                system_prompt="You are a helpful assistant.",
            ),
            "claude-3": ModelConfig(
                api_name="claude-3-opus-20240229",
                client=anthropic_client,
                provider=ModelProvider.ANTHROPIC,
                max_context_tokens=100000,
                system_prompt="You are Claude, a helpful AI assistant.",
            ),
            "human": ModelConfig(
                api_name="human",
                client=None,
                provider=ModelProvider.HUMAN,
                system_prompt="Human participant in the conversation.",
            ),
        }

        for name, config in default_models.items():
            if config.client or config.provider == ModelProvider.HUMAN:
                self.models[name] = config

        for name, config in model_configs.items():
            provider_name = config.get("provider", "").upper()
            if provider_name not in ModelProvider.__members__:
                logger.warning("Unknown provider %s for model %s", provider_name, name)
                continue
            provider = ModelProvider[provider_name]

            role_name = config.get("role", "PRIMARY").upper()
            if role_name not in ModelRole.__members__:
                logger.warning("Unknown role %s for model %s", role_name, name)
                role = ModelRole.PRIMARY
            else:
                role = ModelRole[role_name]

            if provider == ModelProvider.OPENAI and not openai_client:
                logger.warning("Skipping model %s due to missing OpenAI API key", name)
                continue
            if provider == ModelProvider.ANTHROPIC and not anthropic_client:
                logger.warning(
                    "Skipping model %s due to missing Anthropic API key", name
                )
                continue

            client = {
                ModelProvider.OPENAI: openai_client,
                ModelProvider.ANTHROPIC: anthropic_client,
                ModelProvider.HUMAN: None,
                ModelProvider.CLI: None,
            }.get(provider)

            strategy_name = config.get("token_limit_strategy", "TRUNCATE_OLDEST").upper()
            if strategy_name not in TokenLimitStrategy.__members__:
                logger.warning(
                    "Unknown token limit strategy %s for model %s", strategy_name, name
                )
                token_strategy = TokenLimitStrategy.TRUNCATE_OLDEST
            else:
                token_strategy = TokenLimitStrategy[strategy_name]

            self.models[name] = ModelConfig(
                api_name=config.get("api_name", name),
                client=client,
                provider=provider,
                role=role,
                temperature=config.get("temperature", 0.7),
                max_response_tokens=config.get("max_response_tokens", 1024),
                max_context_tokens=config.get("max_context_tokens", 8192),
                token_limit_strategy=token_strategy,
                system_prompt=config.get("system_prompt", ""),
                custom_parameters=config.get("custom_parameters", {}),
            )

    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        return self.models.get(model_name)

    def list_available_models(self) -> List[str]:
        return list(self.models.keys())

    def get_provider(self, provider: ModelProvider) -> ModelProviderInterface:
        return self.providers[provider]
