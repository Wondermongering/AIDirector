from __future__ import annotations
"""Adaptive style management for conversation models."""
from dataclasses import dataclass, field
from typing import List, Dict
from collections import Counter
import re

from conversation_memory import Message, MessageContent


@dataclass
class StyleProfile:
    """Captures observed style preferences."""

    formality: float = 0.5  # 0 informal, 1 formal
    avg_words: float = 0.0
    keywords: Counter = field(default_factory=Counter)
    messages: int = 0


class StyleAdapter:
    """Detects style from history and adjusts prompts."""

    FORMAL_WORDS = {"please", "thank", "regards", "sincerely"}
    INFORMAL_WORDS = {"lol", "haha", "hey", "hi", "!"}

    def __init__(self) -> None:
        self.profile = StyleProfile()

    def update(self, message: Message) -> None:
        if isinstance(message.content, MessageContent):
            text = message.content.text
        else:
            text = str(message.content)
        tokens = re.findall(r"\b\w+\b", text.lower())
        self.profile.messages += 1
        self.profile.avg_words = (
            (self.profile.avg_words * (self.profile.messages - 1)) + len(tokens)
        ) / self.profile.messages
        formal = sum(t in self.FORMAL_WORDS for t in tokens)
        informal = sum(t in self.INFORMAL_WORDS for t in tokens)
        if formal + informal:
            score = formal / (formal + informal)
            self.profile.formality = (
                (self.profile.formality * (self.profile.messages - 1)) + score
            ) / self.profile.messages
        for t in tokens:
            if len(t) > 3:
                self.profile.keywords[t] += 1

    def apply_style(self, base_prompt: str) -> str:
        instructions: List[str] = []
        if self.profile.formality > 0.6:
            instructions.append("Respond in a formal tone.")
        elif self.profile.formality < 0.4:
            instructions.append("Use a friendly, informal tone.")

        if self.profile.avg_words > 25:
            instructions.append("Provide detailed responses.")
        elif self.profile.avg_words and self.profile.avg_words < 15:
            instructions.append("Keep responses concise.")

        if self.profile.keywords:
            top = ", ".join(k for k, _ in self.profile.keywords.most_common(3))
            instructions.append(f"Focus on topics such as {top}.")

        if not instructions:
            return base_prompt
        return base_prompt.rstrip() + "\n" + "\n".join(instructions)
