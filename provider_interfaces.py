"""Abstract interfaces for implementing new AI providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ProviderInterface(ABC):
    """Base interface that all providers should implement."""

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate a response from the provider."""
        raise NotImplementedError


class PluginProvider(ProviderInterface):
    """Simple plugin provider example using an HTTP endpoint."""

    def __init__(self, endpoint: str, api_key: str | None = None) -> None:
        self.endpoint = endpoint
        self.api_key = api_key

    async def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        # Placeholder implementation that would POST to the endpoint.
        # Real implementations should handle networking and errors.
        raise NotImplementedError("Plugin provider not implemented")
