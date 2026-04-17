"""
ChainMind Ollama Provider — Local LLM integration via Ollama HTTP API.

Zero-cost fallback for development, offline, or air-gapped scenarios.
Uses httpx for async HTTP calls — no SDK dependency.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

from chainmind.core.exceptions import LLMProviderError
from chainmind.core.types import LLMRequest, LLMResponse, TokenUsage
from chainmind.llm.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider via HTTP API."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        super().__init__(model=model, api_keys=[])
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def _call_api(self, request: LLMRequest) -> LLMResponse:
        """Call Ollama chat API."""
        try:
            messages = self._build_messages(request)

            payload = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            }

            if request.response_format:
                payload["format"] = "json"

            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)

            usage = TokenUsage(
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider=self.provider_name,
                usage=usage,
                latency_ms=0.0,
            )

        except httpx.ConnectError as e:
            raise LLMProviderError(
                "ollama", f"Cannot connect to Ollama at {self._base_url}: {e}"
            ) from e

        except httpx.HTTPStatusError as e:
            raise LLMProviderError(
                "ollama", f"HTTP {e.response.status_code}: {e}", status_code=e.response.status_code
            ) from e

        except Exception as e:
            raise LLMProviderError("ollama", f"Unexpected error: {e}") from e

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Convert LLMRequest to Ollama message format."""
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream Ollama response tokens."""
        messages = self._build_messages(request)

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as response:
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            # Check if our target model (or a variant) is available
            return any(self._model in m for m in models)
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
