import openai
import anthropic
import datetime
import json
import os
import colorsys
import requests
import tiktoken
import yaml
import argparse
import sys
import signal
import asyncio
import logging
from typing import List, Dict, Generator, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("orchestrator.log"), logging.StreamHandler()]
)
logger = logging.getLogger("orchestrator")

# Create rich console for pretty output
console = Console()

class ModelProvider(Enum):
    OPENAI = auto()
    ANTHROPIC = auto()
    CLI = auto()
    HUMAN = auto()
    
    def __str__(self):
        return self.name.capitalize()

class ModelRole(Enum):
    PRIMARY = "primary"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    MEDIATOR = "mediator"
    CREATIVE = "creative"
    LOGICAL = "logical"
    CUSTOM = "custom"
    
    def __str__(self):
        return self.value.capitalize()

class TokenLimitStrategy(Enum):
    TRUNCATE_OLDEST = auto()
    SUMMARIZE = auto()
    SLIDING_WINDOW = auto()

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
    
    def __post_init__(self):
        if self.role == ModelRole.CUSTOM and not self.system_prompt:
            raise ValueError("Custom role requires a system prompt")

class MessageContent(BaseModel):
    """Content of a message, which can be text or other media."""
    text: str = ""
    media_urls: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text')
    def text_not_empty_if_no_media(cls, v, values):
        if not v and not values.get('media_urls'):
            raise ValueError("Either text or media_urls must be provided")
        return v

@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: Union[str, MessageContent]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def __post_init__(self):
        if isinstance(self.content, str):
            self.content = MessageContent(text=self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        if isinstance(self.content, MessageContent):
            content = self.content.text
        else:
            content = self.content
            
        return {
            "role": self.role,
            "content": content
        }
    
    def to_log_format(self) -> Dict[str, Any]:
        """Convert to format for logging."""
        if isinstance(self.content, MessageContent):
            content_dict = self.content.dict()
        else:
            content_dict = {"text": self.content}
            
        return {
            "role": self.role,
            "content": content_dict,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

class ConversationMemory:
    """Manages conversation history with token tracking and summarization."""
    def __init__(self, max_tokens: int = 8192, strategy: TokenLimitStrategy = TokenLimitStrategy.TRUNCATE_OLDEST):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.messages: List[Message] = []
        self.summaries: List[str] = []
        self.total_tokens = 0
        self.encoder = tiktoken.get_encoding("cl100k_base")  # Default encoding
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        if isinstance(message.content, MessageContent):
            text = message.content.text
        else:
            text = message.content
            
        tokens = len(self.encoder.encode(text))
        
        if self.total_tokens + tokens > self.max_tokens:
            self._apply_token_limit_strategy()
        
        self.messages.append(message)
        self.total_tokens += tokens
    
    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation history."""
        return self.messages
    
    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        """Get messages formatted for API calls."""
        return [message.to_dict() for message in self.messages]
    
    def _apply_token_limit_strategy(self) -> None:
        """Apply the selected strategy to handle token limits."""
        if self.strategy == TokenLimitStrategy.TRUNCATE_OLDEST:
            self._truncate_oldest()
        elif self.strategy == TokenLimitStrategy.SUMMARIZE:
            self._summarize_conversation()
        elif self.strategy == TokenLimitStrategy.SLIDING_WINDOW:
            self._apply_sliding_window()
    
    def _truncate_oldest(self) -> None:
        """Remove oldest messages until under token limit."""
        while self.total_tokens > self.max_tokens * 0.8:  # Keep 80% as buffer
            if not self.messages:
                break
                
            oldest = self.messages.pop(0)
            if isinstance(oldest.content, MessageContent):
                tokens = len(self.encoder.encode(oldest.content.text))
            else:
                tokens = len(self.encoder.encode(oldest.content))
                
            self.total_tokens -= tokens
    
    def _summarize_conversation(self) -> None:
        """Summarize older parts of the conversation."""
        # This would call an AI to summarize older messages
        # For now, just implement truncation as fallback
        self._truncate_oldest()
    
    def _apply_sliding_window(self) -> None:
        """Keep a sliding window of recent messages."""
        # More sophisticated sliding window with key messages retained
        # For now, just implement truncation as fallback
        self._truncate_oldest()

class ConversationLogger:
    """Logs conversation to files with various formats."""
    def __init__(self, log_folder: Path):
        self.log_folder = log_folder
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.log_file = self._create_log_file()
        self.json_log_file = self.log_file.with_suffix(".json")
        self._initialize_json_log()
        
    def _create_log_file(self) -> Path:
        """Create a new log file with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_folder / f"conversation_{timestamp}.log"
    
    def _initialize_json_log(self) -> None:
        """Initialize the JSON log file."""
        with open(self.json_log_file, "w") as f:
            json.dump({"conversation": []}, f)
    
    def log_message(self, actor: str, message: Message) -> None:
        """Log a message to both text and JSON logs."""
        self._log_text(actor, message)
        self._log_json(actor, message)
    
    def _log_text(self, actor: str, message: Message) -> None:
        """Log message in human-readable text format."""
        with open(self.log_file, "a") as f:
            if isinstance(message.content, MessageContent):
                content = message.content.text
            else:
                content = message.content
                
            timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n### {actor} - {timestamp} ###\n{content}\n")
    
    def _log_json(self, actor: str, message: Message) -> None:
        """Log message in structured JSON format."""
        with open(self.json_log_file, "r") as f:
            data = json.load(f)
        
        entry = {
            "actor": actor,
            "message": message.to_log_format()
        }
        
        data["conversation"].append(entry)
        
        with open(self.json_log_file, "w") as f:
            json.dump(data, f, indent=2)

class ColorManager:
    """Manages colors for different actors in the console."""
    def __init__(self):
        self.color_generator = self._generate_distinct_colors()
        self.actor_colors: Dict[str, Tuple[int, int, int]] = {}
    
    def _generate_distinct_colors(self) -> Generator[Tuple[int, int, int], None, None]:
        """Generate visually distinct colors using golden ratio method."""
        hue = 0
        golden_ratio_conjugate = 0.618033988749895
        saturation, value = 0.85, 0.95
        
        while True:
            hue += golden_ratio_conjugate
            hue %= 1
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            yield tuple(int(x * 255) for x in rgb)
    
    def get_color_for_actor(self, actor: str) -> Tuple[int, int, int]:
        """Get a consistent color for an actor."""
        if actor not in self.actor_colors:
            self.actor_colors[actor] = next(self.color_generator)
        return self.actor_colors[actor]
    
    def get_hex_color(self, actor: str) -> str:
        """Get the hex color code for an actor."""
        rgb = self.get_color_for_actor(actor)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

class ConfigurationManager:
    """Manages configuration loading and validation."""
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
            return {}
        
        with open(self.config_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing configuration file: {e}")
                return {}
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations from the config file."""
        return self.config.get("models", {})
    
    def get_orchestration_config(self) -> Dict[str, Any]:
        """Get orchestration configuration."""
        return self.config.get("orchestration", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})

class ModelRegistry:
    """Registry for AI models with initialization of clients."""
    def __init__(self, config_manager: ConfigurationManager):
        load_dotenv()
        self.config_manager = config_manager
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize model clients from configuration."""
        model_configs = self.config_manager.get_model_configs()
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            openai_client = openai.OpenAI(api_key=openai_api_key)
        else:
            openai_client = None
            logger.warning("OpenAI API key not found. OpenAI models will not be available.")
        
        # Initialize Anthropic client
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            anthropic_client = anthropic.Client(api_key=anthropic_api_key)
        else:
            anthropic_client = None
            logger.warning("Anthropic API key not found. Anthropic models will not be available.")
        
        # Initialize default models
        default_models = {
            "gpt-4": ModelConfig(
                api_name="gpt-4",
                client=openai_client,
                provider=ModelProvider.OPENAI,
                max_context_tokens=8192,
                system_prompt="You are a helpful assistant."
            ),
            "claude-3": ModelConfig(
                api_name="claude-3-opus-20240229",
                client=anthropic_client,
                provider=ModelProvider.ANTHROPIC,
                max_context_tokens=100000,
                system_prompt="You are Claude, a helpful AI assistant."
            ),
            "human": ModelConfig(
                api_name="human",
                client=None,
                provider=ModelProvider.HUMAN,
                system_prompt="Human participant in the conversation."
            )
        }
        
        # Add default models
        for name, config in default_models.items():
            if config.client or config.provider == ModelProvider.HUMAN:
                self.models[name] = config
        
        # Add models from configuration
        for name, config in model_configs.items():
            provider_name = config.get("provider", "").upper()
            if provider_name not in ModelProvider.__members__:
                logger.warning(f"Unknown provider {provider_name} for model {name}")
                continue
            
            provider = ModelProvider[provider_name]
            
            role_name = config.get("role", "PRIMARY").upper()
            if role_name not in ModelRole.__members__:
                logger.warning(f"Unknown role {role_name} for model {name}")
                role = ModelRole.PRIMARY
            else:
                role = ModelRole[role_name]
            
            if provider == ModelProvider.OPENAI and not openai_client:
                logger.warning(f"Skipping model {name} due to missing OpenAI API key")
                continue
                
            if provider == ModelProvider.ANTHROPIC and not anthropic_client:
                logger.warning(f"Skipping model {name} due to missing Anthropic API key")
                continue
            
            client = {
                ModelProvider.OPENAI: openai_client,
                ModelProvider.ANTHROPIC: anthropic_client,
                ModelProvider.HUMAN: None,
                ModelProvider.CLI: None
            }.get(provider)
            
            token_limit_strategy_name = config.get("token_limit_strategy", "TRUNCATE_OLDEST").upper()
            if token_limit_strategy_name not in TokenLimitStrategy.__members__:
                logger.warning(f"Unknown token limit strategy {token_limit_strategy_name} for model {name}")
                token_limit_strategy = TokenLimitStrategy.TRUNCATE_OLDEST
            else:
                token_limit_strategy = TokenLimitStrategy[token_limit_strategy_name]
            
            self.models[name] = ModelConfig(
                api_name=config.get("api_name", name),
                client=client,
                provider=provider,
                role=role,
                temperature=config.get("temperature", 0.7),
                max_response_tokens=config.get("max_response_tokens", 1024),
                max_context_tokens=config.get("max_context_tokens", 8192),
                token_limit_strategy=token_limit_strategy,
                system_prompt=config.get("system_prompt", ""),
                custom_parameters=config.get("custom_parameters", {})
            )
    
    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.models.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """List names of all available models."""
        return list(self.models.keys())

class ResponseGenerator:
    """Generates responses from AI models."""
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.world_interface_key = os.getenv("WORLD_INTERFACE_KEY")
    
    async def generate_response(
        self, 
        model_name: str, 
        conversation_memory: ConversationMemory,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response from the specified model."""
        model_config = self.registry.get_model(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Use provided system prompt or default from model config
        effective_system_prompt = system_prompt or model_config.system_prompt
        
        if model_config.provider == ModelProvider.OPENAI:
            return await self._generate_openai_response(model_config, conversation_memory, effective_system_prompt)
        elif model_config.provider == ModelProvider.ANTHROPIC:
            return await self._generate_anthropic_response(model_config, conversation_memory, effective_system_prompt)
        elif model_config.provider == ModelProvider.CLI:
            return await self._generate_cli_response(model_config, conversation_memory, effective_system_prompt)
        elif model_config.provider == ModelProvider.HUMAN:
            return self._generate_human_response()
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    async def _generate_openai_response(
        self, 
        model_config: ModelConfig, 
        conversation_memory: ConversationMemory,
        system_prompt: str
    ) -> str:
        """Generate a response using OpenAI API."""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_memory.get_formatted_messages())
            
            response = await asyncio.to_thread(
                model_config.client.chat.completions.create,
                model=model_config.api_name,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_response_tokens,
                **model_config.custom_parameters
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return f"[Error generating response: {str(e)}]"
    
    async def _generate_anthropic_response(
        self, 
        model_config: ModelConfig, 
        conversation_memory: ConversationMemory,
        system_prompt: str
    ) -> str:
        """Generate a response using Anthropic API."""
        try:
            messages = conversation_memory.get_formatted_messages()
            
            response = await asyncio.to_thread(
                model_config.client.messages.create,
                model=model_config.api_name,
                system=system_prompt,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_response_tokens,
                **model_config.custom_parameters
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return f"[Error generating response: {str(e)}]"
    
    async def _generate_cli_response(
        self, 
        model_config: ModelConfig, 
        conversation_memory: ConversationMemory,
        system_prompt: str
    ) -> str:
        """Generate a response using a local CLI API."""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_memory.get_formatted_messages())
            
            response = await asyncio.to_thread(
                lambda: requests.post(
                    "http://localhost:3000/v1/chat/completions",
                    json={"messages": messages},
                    headers={
                        "Authorization": f"Bearer {self.world_interface_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
            )
            
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating CLI response: {e}")
            return f"[Error generating response: {str(e)}]"
    
    def _generate_human_response(self) -> str:
        """Get response from human user input."""
        return Prompt.ask("\n[Human Input]")

class AIOrchestrator:
    """Orchestrates conversations between multiple AI models."""
    def __init__(
        self, 
        models: List[str], 
        max_turns: int = 10, 
        log_folder: str = "OrchestratorLogs",
        config_path: Optional[str] = None
    ):
        self.models = models
        self.max_turns = max_turns
        self.turn = 0
        self.config_manager = ConfigurationManager(Path(config_path) if config_path else None)
        self.registry = ModelRegistry(self.config_manager)
        self.response_generator = ResponseGenerator(self.registry)
        self.logger = ConversationLogger(Path(log_folder))
        self.color_manager = ColorManager()
        self.conversation_memories: Dict[str, ConversationMemory] = {}
        self.shared_memory = ConversationMemory(max_tokens=100000)  # Shared conversation history
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        # Initialize conversation memories for each model
        self._initialize_memories()
    
    def _initialize_memories(self) -> None:
        """Initialize conversation memories for each model."""
        for model_name in self.models:
            model_config = self.registry.get_model(model_name)
            if model_config:
                self.conversation_memories[model_name] = ConversationMemory(
                    max_tokens=model_config.max_context_tokens,
                    strategy=model_config.token_limit_strategy
                )
    
    def _handle_interrupt(self, sig, frame) -> None:
        """Handle interrupt signals."""
        console.print("\n[bold red]Received interrupt signal. Shutting down...[/bold red]")
        self.running = False
    
    def display_message(self, actor: str, message: Message) -> None:
        """Display a message in the console with color and formatting."""
        if isinstance(message.content, MessageContent):
            content = message.content.text
        else:
            content = message.content
            
        color = self.color_manager.get_hex_color(actor)
        console.print(Panel(Markdown(content), title=f"[bold]{actor}[/bold]", border_style=color))
    
    async def run_conversation(self) -> None:
        """Run the orchestrated conversation."""
        console.print("[bold]Conversation Initialized![/bold]")
        
        # Add system messages to each model's memory
        await self._initialize_conversation()
        
        while self.turn < self.max_turns and self.running:
            console.print(f"\n[bold]Turn {self.turn + 1}/{self.max_turns}[/bold]")
            
            for model_name in self.models:
                model_config = self.registry.get_model(model_name)
                if not model_config:
                    logger.warning(f"Model {model_name} not found in registry. Skipping.")
                    continue
                
                actor_name = f"{model_config.provider} ({model_name})"
                
                try:
                    # Generate response
                    response_text = await self.response_generator.generate_response(
                        model_name, 
                        self.shared_memory,
                        model_config.system_prompt
                    )
                    
                    # Create message object
                    response_message = Message(
                        role="assistant" if model_config.provider != ModelProvider.HUMAN else "user",
                        content=response_text,
                        metadata={"model": model_name, "role": model_config.role.value}
                    )
                    
                    # Add to conversation memories
                    self.shared_memory.add_message(response_message)
                    self.conversation_memories[model_name].add_message(response_message)
                    
                    # Display and log the message
                    self.display_message(actor_name, response_message)
                    self.logger.log_message(actor_name, response_message)
                    
                    # Check if conversation should continue after each response
                    if not self.running:
                        break
                    
                except Exception as e:
                    logger.error(f"Error with {actor_name}: {str(e)}")
                    console.print(f"[bold red]Error with {actor_name}:[/bold red] {str(e)}")
                    continue
            
            self.turn += 1
            
            # After each full turn, check if the user wants to continue
            if self.running and self.turn < self.max_turns:
                if not Confirm.ask("\nContinue to next turn?", default=True):
                    self.running = False
        
        console.print("\n[bold green]Conversation Complete.[/bold green]")
    
    async def _initialize_conversation(self) -> None:
        """Initialize the conversation with a starting message."""
        # Check if human is a participant
        has_human = "human" in self.models
        
        if has_human:
            # If human is participating, let them start
            human_starter = Message(
                role="user",
                content=Prompt.ask("\n[bold]Start the conversation[/bold]")
            )
            self.shared_memory.add_message(human_starter)
            self.conversation_memories["human"].add_message(human_starter)
            self.logger.log_message("Human", human_starter)
        else:
            # Otherwise, use a system message to start
            starter_message = Message(
                role="system",
                content="The conversation begins now. Each participant should introduce themselves and their role."
            )
            self.shared_memory.add_message(starter_message)
            for model_name in self.models:
                self.conversation_memories[model_name].add_message(starter_message)
            self.logger.log_message("System", starter_message)

async def main():
    """Main entry point for the AI Orchestrator."""
    parser = argparse.ArgumentParser(description="AI Conversation Orchestrator")
    parser.add_argument("--models", nargs="+", default=["gpt-4", "claude-3"], help="Models to include in conversation")
    parser.add_argument("--turns", type=int, default=5, help="Maximum number of conversation turns")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-folder", type=str, default="OrchestratorLogs", help="Folder for conversation logs")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = AIOrchestrator(
        models=args.models,
        max_turns=args.turns,
        log_folder=args.log_folder,
        config_path=args.config
    )
    
    # List available models if requested
    if args.list_models:
        console.print("[bold]Available Models:[/bold]")
        for model_name in orchestrator.registry.list_available_models():
            model = orchestrator.registry.get_model(model_name)
            console.print(f"- {model_name} ({model.provider})")
        return
    
    # Run the conversation
    await orchestrator.run_conversation()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
        sys.exit(0)
