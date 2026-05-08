"""
chainmind/llm/openai_provider.py
OpenAI / GPT-4 provider — research baseline only.

Used for:
- GPT-4 direct baseline (no tools) in ChainMind-Bench Table 1
- ChainMind+GPT-4 baseline in Table 3
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from chainmind.core.interfaces import ILLMProvider
from chainmind.core.types import LLMMessage, LLMRequest, LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIProvider(ILLMProvider):
    """OpenAI API provider with key rotation."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_keys: list[str] | None = None,
    ):
        self._model = model
        self._api_keys = api_keys or []
        self._current_key_idx = 0

    def _get_client(self):
        """Get an OpenAI client with the next available key."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai>=1.40.0")

        if not self._api_keys:
            raise ValueError("No OpenAI API keys configured")

        key = self._api_keys[self._current_key_idx % len(self._api_keys)]
        self._current_key_idx = (self._current_key_idx + 1) % len(self._api_keys)
        return openai.AsyncOpenAI(api_key=key)

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Convert internal messages to OpenAI format."""
        msgs = []
        if request.system_prompt:
            msgs.append({"role": "system", "content": request.system_prompt})
        for m in request.messages:
            msgs.append({"role": m.role if m.role != "tool" else "user", "content": m.content})
        return msgs

    async def generate(self, request: LLMRequest) -> LLMResponse:
        client = self._get_client()
        start = time.perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": self._build_messages(request),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences
            if request.response_format:
                kwargs["response_format"] = request.response_format

            response = await client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            content = response.choices[0].message.content or ""
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider="openai",
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            from chainmind.core.exceptions import LLMProviderError
            raise LLMProviderError(f"OpenAI generation failed: {e}") from e

    async def generate_structured(
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> LLMResponse:
        """Use JSON mode for structured output."""
        modified = request.model_copy(
            update={"response_format": {"type": "json_object"}}
        )
        return await self.generate(modified)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        client = self._get_client()
        msgs = self._build_messages(request)
        async with client.chat.completions.stream(
            model=self._model,
            messages=msgs,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            models = await client.models.list()
            return any(self._model in m.id for m in models.data)
        except Exception:
            return False
