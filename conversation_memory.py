"""Utilities for conversation messages and memory management."""
import asyncio
import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Union

import tiktoken
from pydantic import BaseModel, Field, validator

from model_registry import ConfigurationManager, ModelRegistry, ModelRole
from response_generator import ResponseGenerator


class TokenLimitStrategy(Enum):
    TRUNCATE_OLDEST = auto()
    SUMMARIZE = auto()
    SLIDING_WINDOW = auto()


class MessageContent(BaseModel):
    """Content of a message, which can be text or other media."""

    text: str = ""
    media_urls: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("text")
    def text_not_empty_if_no_media(cls, v, values):
        if not v and not values.get("media_urls"):
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
        if isinstance(self.content, MessageContent):
            content = self.content.text
        else:
            content = self.content
        return {"role": self.role, "content": content}

    def to_log_format(self) -> Dict[str, Any]:
        if isinstance(self.content, MessageContent):
            content_dict = self.content.dict()
        else:
            content_dict = {"text": self.content}
        return {
            "role": self.role,
            "content": content_dict,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationMemory:
    """Manages conversation history with token tracking and summarization."""

    def __init__(self, max_tokens: int = 8192, strategy: TokenLimitStrategy = TokenLimitStrategy.TRUNCATE_OLDEST, encoder_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.strategy = strategy
        self.messages: List[Message] = []
        self.summaries: List[str] = []
        self.total_tokens = 0
        self.encoder = tiktoken.get_encoding(encoder_name)

    def add_message(self, message: Message) -> None:
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
        return self.messages

    def get_formatted_messages(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    def _apply_token_limit_strategy(self) -> None:
        if self.strategy == TokenLimitStrategy.TRUNCATE_OLDEST:
            self._truncate_oldest()
        elif self.strategy == TokenLimitStrategy.SUMMARIZE:
            self._summarize_conversation()
        elif self.strategy == TokenLimitStrategy.SLIDING_WINDOW:
            self._apply_sliding_window()

    def _truncate_oldest(self) -> None:
        while self.total_tokens > self.max_tokens * 0.8:
            if not self.messages:
                break
            oldest = self.messages.pop(0)
            if isinstance(oldest.content, MessageContent):
                tokens = len(self.encoder.encode(oldest.content.text))
            else:
                tokens = len(self.encoder.encode(oldest.content))
            self.total_tokens -= tokens

    def _summarize_conversation(self) -> None:
        if len(self.messages) < 2:
            return

        half_index = len(self.messages) // 2
        to_summarize = self.messages[:half_index]
        summary_text = self._get_summary_from_model(to_summarize)
        summary_message = Message(role="system", content=summary_text)

        self.summaries.append(summary_text)
        remaining = self.messages[half_index:]
        self.messages = [summary_message] + remaining

        # Recalculate tokens from scratch
        self.total_tokens = 0
        for msg in self.messages:
            if isinstance(msg.content, MessageContent):
                text = msg.content.text
            else:
                text = msg.content
            self.total_tokens += len(self.encoder.encode(text))

    def _apply_sliding_window(self) -> None:
        target_tokens = int(self.max_tokens * 0.8)
        while self.total_tokens > target_tokens:
            if not self.messages:
                break
            oldest = self.messages.pop(0)
            if isinstance(oldest.content, MessageContent):
                tokens = len(self.encoder.encode(oldest.content.text))
            else:
                tokens = len(self.encoder.encode(oldest.content))
            self.total_tokens -= tokens

    def _get_summary_from_model(self, messages: List[Message]) -> str:
        registry = ModelRegistry(ConfigurationManager())
        generator = ResponseGenerator(registry)
        summarizer_name = None
        for name, config in registry.models.items():
            if config.role == ModelRole.SUMMARIZER:
                summarizer_name = name
                break
        if not summarizer_name:
            summarizer_name = next(iter(registry.models.keys()))

        temp_memory = ConversationMemory(max_tokens=self.max_tokens)
        for msg in messages:
            temp_memory.add_message(msg)

        prompt = "Provide a concise summary of the following conversation."
        try:
            return asyncio.run(
                generator.generate_response(summarizer_name, temp_memory, prompt)
            )
        except Exception as e:  # pragma: no cover - network failures
            return f"[Summary error: {e}]"
