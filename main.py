"""Entry point for the AI Orchestrator CLI."""
from cli import run_cli, AIOrchestrator
from conversation_memory import (
    ConversationMemory,
    Message,
    MessageContent,
    TokenLimitStrategy,
)
from model_registry import ModelRegistry, ConfigurationManager, ModelProvider, ModelRole, ModelConfig
from response_generator import ResponseGenerator
from semantic_memory import SemanticMemory
from plugin_manager import PluginManager
from turn_taking import DialogicTurnCoordinator


if __name__ == "__main__":
    run_cli()

