from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
from dotenv import load_dotenv
import openai
import anthropic
import requests
from typing import List, Dict, Generator, Tuple, Optional

# Load environment variables from .env file
load_dotenv()

class ModelProvider(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    CLI = "CLI"

@dataclass
class ModelConfig:
    api_name: str
    client: any
    provider: ModelProvider

@dataclass
class Message:
    role: str
    content: str

class ConversationLogger:
    def __init__(self, log_folder: Path):
        self.log_folder = log_folder
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.log_file = self._create_log_file()
        
    def _create_log_file(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.log_folder / f"conversation_{timestamp}.log"
        
    def log_message(self, actor: str, message: str) -> None:
        with open(self.log_file, "a") as f:
            f.write(f"\n### {actor} ###\n{message}\n")

class ColorManager:
    def __init__(self):
        self.color_generator = self._generate_distinct_colors()
        self.actor_colors: Dict[str, str] = {}
        
    def _generate_distinct_colors(self) -> Generator[Tuple[int, int, int], None, None]:
        hue = 0
        golden_ratio_conjugate = 0.618033988749895
        while True:
            hue += golden_ratio_conjugate
            hue %= 1
            rgb = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
            yield tuple(int(x * 255) for x in rgb)
            
    def get_color_for_actor(self, actor: str) -> str:
        if actor not in self.actor_colors:
            rgb = next(self.color_generator)
            self.actor_colors[actor] = f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        return self.actor_colors[actor]

class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = self._initialize_models()
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        return {
            "gpt-4": ModelConfig("gpt-4", openai_client, ModelProvider.OPENAI),
            "claude-3": ModelConfig("claude-3", anthropic_client, ModelProvider.ANTHROPIC),
            "world-interface": ModelConfig("world-interface", None, ModelProvider.CLI)
        }
        
    def get_model(self, model_name: str) -> Optional[ModelConfig]:
        return self.models.get(model_name)

class ResponseGenerator:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.world_interface_key = os.getenv("WORLD_INTERFACE_KEY")
        
    def generate_response(self, model_name: str, context: List[Message], system_prompt: str = "") -> str:
        model_config = self.registry.get_model(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
            
        if model_config.provider == ModelProvider.OPENAI:
            return self._generate_openai_response(model_config, context)
        elif model_config.provider == ModelProvider.ANTHROPIC:
            return self._generate_anthropic_response(model_config, context)
        elif model_config.provider == ModelProvider.CLI:
            return self._generate_cli_response(context)
            
    def _generate_openai_response(self, model_config: ModelConfig, context: List[Message]) -> str:
        response = model_config.client.chat.completions.create(
            model=model_config.api_name,
            messages=[msg.__dict__ for msg in context],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
        
    def _generate_anthropic_response(self, model_config: ModelConfig, context: List[Message]) -> str:
        messages = [{"role": msg.role, "content": msg.content} for msg in context]
        message = model_config.client.messages.create(
            model=model_config.api_name,
            messages=messages,
            temperature=1.0,
            max_tokens=1024
        )
        return message.content[0].text
        
    def _generate_cli_response(self, context: List[Message]) -> str:
        response = requests.post(
            "http://localhost:3000/v1/chat/completions",
            json={"messages": [msg.__dict__ for msg in context]},
            headers={
                "Authorization": f"Bearer {self.world_interface_key}",
                "Content-Type": "application/json"
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

class AIOrchestrator:
    def __init__(self, models: List[str], max_turns: int = 10, log_folder: str = "OrchestratorLogs"):
        self.models = models
        self.max_turns = max_turns
        self.turn = 0
        self.conversation_history: List[Message] = []
        self.logger = ConversationLogger(Path(log_folder))
        self.color_manager = ColorManager()
        self.registry = ModelRegistry()
        self.response_generator = ResponseGenerator(self.registry)
        
    def display_message(self, actor: str, message: str) -> None:
        color = self.color_manager.get_color_for_actor(actor)
        bold, reset = "\033[1m", "\033[0m"
        print(f"\n{bold}{color}{actor}:{reset} {message}")
        
    def run_conversation(self) -> None:
        print("Conversation Initialized!")
        
        while self.turn < self.max_turns:
            for i, model in enumerate(self.models):
                model_config = self.registry.get_model(model)
                if not model_config:
                    continue
                    
                actor_name = f"{model_config.provider.value} Model {i + 1}"
                system_prompt = f"Role of {actor_name} in this context"
                
                try:
                    response = self.response_generator.generate_response(
                        model, self.conversation_history, system_prompt
                    )
                    
                    self.conversation_history.append(Message("assistant", response))
                    self.display_message(actor_name, response)
                    self.logger.log_message(actor_name, response)
                    
                except Exception as e:
                    print(f"Error with {actor_name}: {str(e)}")
                    continue
                    
            self.turn += 1
            
        print("\nConversation Complete.")

if __name__ == "__main__":
    orchestrator = AIOrchestrator(
        models=["gpt-4", "claude-3", "world-interface"],
        max_turns=5
    )
    orchestrator.run_conversation()
