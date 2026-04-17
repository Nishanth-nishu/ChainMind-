"""
ChainMind Base LLM Provider — Abstract base for all LLM providers.

Implements common logic: request validation, response normalization,
timing instrumentation. Concrete providers only implement _call_api().
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any, AsyncIterator

from chainmind.core.interfaces import ILLMProvider
from chainmind.core.types import LLMRequest, LLMResponse, TokenUsage


class BaseLLMProvider(ILLMProvider):
    """Base class for LLM providers with common instrumentation."""

    def __init__(self, model: str, api_keys: list[str] | None = None):
        self._model = model
        self._api_keys = api_keys or []
        self._current_key_index = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def available_keys(self) -> int:
        return len(self._api_keys)

    def _get_current_key(self) -> str | None:
        """Get the current API key from the pool."""
        if not self._api_keys:
            return None
        return self._api_keys[self._current_key_index % len(self._api_keys)]

    def _rotate_key(self) -> str | None:
        """Rotate to the next API key in the pool (round-robin)."""
        if not self._api_keys:
            return None
        self._current_key_index = (self._current_key_index + 1) % len(self._api_keys)
        return self._api_keys[self._current_key_index]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate with timing instrumentation."""
        start = time.perf_counter()
        response = await self._call_api(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = elapsed_ms
        response.provider = self.provider_name
        response.model = self._model
        return response

    async def generate_structured(
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> LLMResponse:
        """Generate structured output — default falls back to generate with schema in prompt."""
        if request.response_format is None:
            request.response_format = schema
        return await self.generate(request)

    @abstractmethod
    async def _call_api(self, request: LLMRequest) -> LLMResponse:
        """Provider-specific API call. Must be implemented by subclasses."""
        ...

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream tokens — must be implemented by subclasses."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Health check — must be implemented by subclasses."""
        ...
