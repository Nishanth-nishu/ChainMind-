"""
ChainMind Gemini Provider — Google Generative AI integration.

Supports structured output via JSON schema (Amazon Nova constrained decoding pattern).
Round-robin key rotation with automatic failover on 429 errors.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import google.generativeai as genai

from chainmind.core.exceptions import LLMProviderError, LLMQuotaExhaustedError
from chainmind.core.types import LLMRequest, LLMResponse, TokenUsage
from chainmind.llm.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider with multi-key round-robin."""

    def __init__(self, model: str, api_keys: list[str]):
        super().__init__(model=model, api_keys=api_keys)
        self._clients: dict[str, genai.GenerativeModel] = {}
        self._configure_clients()

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _configure_clients(self) -> None:
        """Pre-configure a client per API key for fast rotation."""
        for i, key in enumerate(self._api_keys):
            try:
                genai.configure(api_key=key)
                self._clients[key] = genai.GenerativeModel(self._model)
            except Exception as e:
                logger.warning(f"Failed to configure Gemini key #{i}: {e}")

    def _get_client(self) -> genai.GenerativeModel:
        """Get client for the current API key."""
        key = self._get_current_key()
        if key is None or key not in self._clients:
            raise LLMProviderError("gemini", "No valid API keys configured")
        # Re-configure with current key before use
        genai.configure(api_key=key)
        return self._clients[key]

    async def _call_api(self, request: LLMRequest) -> LLMResponse:
        """Call Gemini API with automatic key rotation on quota errors."""
        last_error: Exception | None = None
        tried_keys = 0

        while tried_keys < len(self._api_keys):
            try:
                client = self._get_client()

                # Build message content
                contents = self._build_contents(request)

                # Generation config
                gen_config = genai.GenerationConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_tokens,
                )

                # Add response schema if structured output requested
                if request.response_format:
                    gen_config.response_mime_type = "application/json"

                response = client.generate_content(
                    contents=contents,
                    generation_config=gen_config,
                )

                # Extract usage
                usage = TokenUsage(
                    prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                    completion_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
                    total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
                )

                return LLMResponse(
                    content=response.text or "",
                    model=self._model,
                    provider=self.provider_name,
                    usage=usage,
                    latency_ms=0.0,  # Set by base class
                )

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                if "429" in error_str or "quota" in error_str or "rate" in error_str:
                    logger.warning(
                        f"Gemini key #{self._current_key_index} rate-limited, rotating..."
                    )
                    self._rotate_key()
                    tried_keys += 1
                    continue
                else:
                    raise LLMProviderError(
                        "gemini", f"API error: {e}", status_code=getattr(e, "code", None)
                    ) from e

        raise LLMQuotaExhaustedError(
            "gemini", f"All {len(self._api_keys)} API keys exhausted: {last_error}"
        )

    def _build_contents(self, request: LLMRequest) -> list[dict[str, Any]] | str:
        """Convert LLMRequest messages to Gemini content format."""
        parts = []

        # System prompt as first part
        if request.system_prompt:
            parts.append(request.system_prompt)

        # Conversation messages
        for msg in request.messages:
            prefix = ""
            if msg.role == "user":
                prefix = ""
            elif msg.role == "assistant":
                prefix = ""
            elif msg.role == "tool":
                prefix = f"[Tool Result] "
            parts.append(f"{prefix}{msg.content}")

        return "\n\n".join(parts)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream Gemini response tokens."""
        client = self._get_client()
        contents = self._build_contents(request)

        gen_config = genai.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        response = client.generate_content(
            contents=contents,
            generation_config=gen_config,
            stream=True,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def health_check(self) -> bool:
        """Quick health check with minimal token usage."""
        try:
            client = self._get_client()
            response = client.generate_content(
                "Reply with 'ok'",
                generation_config=genai.GenerationConfig(max_output_tokens=5),
            )
            return bool(response.text)
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False
