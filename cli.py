"""Command-line interface for running AI Orchestrator conversations."""
import argparse
import asyncio
import logging
import signal
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from dotenv import load_dotenv

from conversation_memory import ConversationMemory, Message, MessageContent
from semantic_memory import SemanticMemory
from model_registry import ConfigurationManager, ModelRegistry, ModelProvider
from response_generator import ResponseGenerator
from plugin_manager import PluginManager

logger = logging.getLogger(__name__)
console = Console()


class ConversationLogger:
    """Logs conversation to files."""

    def __init__(self, log_folder: Path) -> None:
        self.log_folder = log_folder
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.log_file = self._create_log_file()
        self.json_log_file = self.log_file.with_suffix(".json")
        self._initialize_json_log()

    def _create_log_file(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_folder / f"conversation_{timestamp}.log"

    def _initialize_json_log(self) -> None:
        with open(self.json_log_file, "w") as f:
            json.dump({"conversation": []}, f)

    def log_message(self, actor: str, message: Message) -> None:
        self._log_text(actor, message)
        self._log_json(actor, message)

    def _log_text(self, actor: str, message: Message) -> None:
        with open(self.log_file, "a") as f:
            if isinstance(message.content, MessageContent):
                content = message.content.text
            else:
                content = message.content
            timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n### {actor} - {timestamp} ###\n{content}\n")

    def _log_json(self, actor: str, message: Message) -> None:
        with open(self.json_log_file, "r") as f:
            data = json.load(f)
        data["conversation"].append({"actor": actor, "message": message.to_log_format()})
        with open(self.json_log_file, "w") as f:
            json.dump(data, f, indent=2)


class ColorManager:
    """Manage distinct colors for actors."""

    def __init__(self) -> None:
        self.color_generator = self._generate_distinct_colors()
        self.actor_colors: Dict[str, tuple[int, int, int]] = {}

    def _generate_distinct_colors(self):
        import colorsys

        hue = 0
        golden_ratio_conjugate = 0.618033988749895
        saturation, value = 0.85, 0.95
        while True:
            hue += golden_ratio_conjugate
            hue %= 1
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            yield tuple(int(x * 255) for x in rgb)

    def get_color_for_actor(self, actor: str) -> tuple[int, int, int]:
        if actor not in self.actor_colors:
            self.actor_colors[actor] = next(self.color_generator)
        return self.actor_colors[actor]

    def get_hex_color(self, actor: str) -> str:
        rgb = self.get_color_for_actor(actor)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class AIOrchestrator:
    """Orchestrates conversations between multiple AI models."""

    def __init__(
        self,
        models: List[str],
        max_turns: int = 10,
        log_folder: str = "OrchestratorLogs",
        config_path: str | None = None,
    ) -> None:
        self.models = models
        self.max_turns = max_turns
        self.turn = 0
        self.config_manager = ConfigurationManager(Path(config_path) if config_path else None)
        self.registry = ModelRegistry(self.config_manager)
        self.response_generator = ResponseGenerator(self.registry)
        self.plugins = PluginManager(self.config_manager)
        self.logger = ConversationLogger(Path(log_folder))
        self.color_manager = ColorManager()
        self.semantic_memory = SemanticMemory()
        self.conversation_memories: Dict[str, ConversationMemory] = {}
        self.shared_memory = ConversationMemory(max_tokens=100000)
        self.running = True

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self._initialize_memories()
        self.plugins.run_hook("on_init", orchestrator=self)

    def _initialize_memories(self) -> None:
        for model_name in self.models:
            model_config = self.registry.get_model(model_name)
            if model_config:
                self.conversation_memories[model_name] = ConversationMemory(
                    max_tokens=model_config.max_context_tokens,
                    strategy=model_config.token_limit_strategy,
                )

    def _handle_interrupt(self, sig, frame) -> None:  # pragma: no cover - CLI utility
        console.print("\n[bold red]Received interrupt signal. Shutting down...[/bold red]")
        self.running = False

    def display_message(self, actor: str, message: Message) -> None:
        if isinstance(message.content, MessageContent):
            content = message.content.text
        else:
            content = message.content
        color = self.color_manager.get_hex_color(actor)
        console.print(Panel(Markdown(content), title=f"[bold]{actor}[/bold]", border_style=color))

    async def run_conversation(self) -> None:
        console.print("[bold]Conversation Initialized![/bold]")
        await self._initialize_conversation()
        self.plugins.run_hook("conversation_start", orchestrator=self)

        while self.turn < self.max_turns and self.running:
            console.print(f"\n[bold]Turn {self.turn + 1}/{self.max_turns}[/bold]")
            for model_name in self.models:
                model_config = self.registry.get_model(model_name)
                if not model_config:
                    logger.warning("Model %s not found in registry. Skipping.", model_name)
                    continue
                actor_name = f"{model_config.provider} ({model_name})"
                try:
                    last_msg = self.shared_memory.get_messages()[-1]
                    if isinstance(last_msg.content, MessageContent):
                        query = last_msg.content.text
                    else:
                        query = last_msg.content
                    context_items = self.semantic_memory.retrieve_relevant(query, top_k=3)
                    context_text = "\n".join(item["text"] for item in context_items)
                    prompt = model_config.system_prompt
                    if context_text:
                        prompt = f"{prompt}\nRelevant context:\n{context_text}"
                    response_text = await self.response_generator.generate_response(
                        model_name, self.shared_memory, prompt
                    )
                    response_message = Message(
                        role="assistant" if model_config.provider != ModelProvider.HUMAN else "user",
                        content=response_text,
                        metadata={"model": model_name, "role": model_config.role.value},
                    )
                    self.shared_memory.add_message(response_message)
                    self.conversation_memories[model_name].add_message(response_message)
                    self.semantic_memory.add_memory(response_message.content.text, {"actor": model_name})
                    self.display_message(actor_name, response_message)
                    self.logger.log_message(actor_name, response_message)
                    self.plugins.run_hook(
                        "message", actor=actor_name, message=response_message
                    )
                    if not self.running:
                        break
                except Exception as e:  # pragma: no cover - runtime path
                    logger.error("Error with %s: %s", actor_name, e)
                    console.print(f"[bold red]Error with {actor_name}:[/bold red] {e}")
                    continue
            self.turn += 1
            if self.running and self.turn < self.max_turns:
                if not Confirm.ask("\nContinue to next turn?", default=True):
                    self.running = False
        console.print("\n[bold green]Conversation Complete.[/bold green]")
        self.plugins.run_hook("conversation_end", orchestrator=self)

    async def _initialize_conversation(self) -> None:
        has_human = "human" in self.models
        if has_human:
            human_starter = Message(role="user", content=Prompt.ask("\n[bold]Start the conversation[/bold]"))
            self.shared_memory.add_message(human_starter)
            self.conversation_memories["human"].add_message(human_starter)
            self.semantic_memory.add_memory(human_starter.content.text, {"actor": "human"})
            self.logger.log_message("Human", human_starter)
            self.plugins.run_hook("message", actor="Human", message=human_starter)
        else:
            starter_message = Message(
                role="system",
                content="The conversation begins now. Each participant should introduce themselves and their role.",
            )
            self.shared_memory.add_message(starter_message)
            for model_name in self.models:
                self.conversation_memories[model_name].add_message(starter_message)
                self.semantic_memory.add_memory(starter_message.content.text, {"actor": "system"})
            self.logger.log_message("System", starter_message)
            self.plugins.run_hook("message", actor="System", message=starter_message)


def run_cli() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI Conversation Orchestrator")
    parser.add_argument("--models", nargs="+", default=["gpt-4", "claude-3"], help="Models to include in conversation")
    parser.add_argument("--turns", type=int, default=5, help="Maximum number of conversation turns")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-folder", type=str, default="OrchestratorLogs", help="Folder for conversation logs")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    args = parser.parse_args()

    orchestrator = AIOrchestrator(
        models=args.models,
        max_turns=args.turns,
        log_folder=args.log_folder,
        config_path=args.config,
    )

    if args.list_models:
        console.print("[bold]Available Models:[/bold]")
        for model_name in orchestrator.registry.list_available_models():
            model = orchestrator.registry.get_model(model_name)
            console.print(f"- {model_name} ({model.provider})")
        return

    asyncio.run(orchestrator.run_conversation())

