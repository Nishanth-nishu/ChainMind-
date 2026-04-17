"""
ChainMind OpenAI Provider — OpenAI API integration.

Compatible fallback when Gemini quota is exhausted.
Supports structured outputs via response_format.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, APIError, RateLimitError

from chainmind.core.exceptions import LLMProviderError, LLMQuotaExhaustedError
from chainmind.core.types import LLMRequest, LLMResponse, TokenUsage
from chainmind.llm.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider with multi-key round-robin."""

    def __init__(self, model: str, api_keys: list[str]):
        super().__init__(model=model, api_keys=api_keys)
        self._clients: dict[str, AsyncOpenAI] = {}
        self._configure_clients()

    @property
    def provider_name(self) -> str:
        return "openai"

    def _configure_clients(self) -> None:
        """Pre-configure an async client per API key."""
        for key in self._api_keys:
            self._clients[key] = AsyncOpenAI(api_key=key)

    def _get_client(self) -> AsyncOpenAI:
        """Get client for the current API key."""
        key = self._get_current_key()
        if key is None or key not in self._clients:
            raise LLMProviderError("openai", "No valid API keys configured")
        return self._clients[key]

    async def _call_api(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API with automatic key rotation on rate limits."""
        last_error: Exception | None = None
        tried_keys = 0

        while tried_keys < max(len(self._api_keys), 1):
            try:
                client = self._get_client()
                messages = self._build_messages(request)

                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                }

                if request.response_format:
                    kwargs["response_format"] = {"type": "json_object"}

                if request.stop_sequences:
                    kwargs["stop"] = request.stop_sequences

                response = await client.chat.completions.create(**kwargs)

                choice = response.choices[0]
                usage_data = response.usage

                usage = TokenUsage(
                    prompt_tokens=usage_data.prompt_tokens if usage_data else 0,
                    completion_tokens=usage_data.completion_tokens if usage_data else 0,
                    total_tokens=usage_data.total_tokens if usage_data else 0,
                )

                return LLMResponse(
                    content=choice.message.content or "",
                    model=self._model,
                    provider=self.provider_name,
                    usage=usage,
                    latency_ms=0.0,
                )

            except RateLimitError as e:
                logger.warning(
                    f"OpenAI key #{self._current_key_index} rate-limited, rotating..."
                )
                self._rotate_key()
                tried_keys += 1
                last_error = e
                continue

            except APIError as e:
                raise LLMProviderError(
                    "openai", f"API error: {e}", status_code=e.status_code
                ) from e

            except Exception as e:
                raise LLMProviderError("openai", f"Unexpected error: {e}") from e

        raise LLMQuotaExhaustedError(
            "openai", f"All {len(self._api_keys)} API keys exhausted: {last_error}"
        )

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Convert LLMRequest to OpenAI message format."""
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream OpenAI response tokens."""
        client = self._get_client()
        messages = self._build_messages(request)

        stream = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def health_check(self) -> bool:
        """Quick health check."""
        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Reply with 'ok'"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
