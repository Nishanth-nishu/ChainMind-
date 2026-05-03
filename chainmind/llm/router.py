"""
ChainMind LLM Router — Local-only inference with optional OpenAI research baseline.

Primary provider: Local vLLM server (GPU-accelerated, zero API cost, full privacy).
Optional baseline: OpenAI GPT-4 (for research comparison only; requires API key).

Gemini and Ollama have been removed. ChainMind runs exclusively on local LLMs.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, AsyncIterator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from chainmind.config.constants import CircuitState
from chainmind.config.settings import LLMProvider, Settings
from chainmind.core.exceptions import (
    LLMAllProvidersFailedError,
    LLMCircuitOpenError,
    LLMError,
    LLMProviderError,
    LLMQuotaExhaustedError,
)
from chainmind.core.interfaces import ILLMProvider
from chainmind.core.types import LLMRequest, LLMResponse
from chainmind.llm.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Routes LLM requests across providers with fault tolerance.

    Architecture:
    ┌─────────────────────────────────────────────┐
    │              LLM Router                      │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
    │  │ Gemini  │  │ OpenAI  │  │ Ollama  │     │
    │  │ [CB]    │  │ [CB]    │  │ [CB]    │     │
    │  │ key1    │  │ key1    │  │ local   │     │
    │  │ key2    │  │ key2    │  │         │     │
    │  │ key3    │  │         │  │         │     │
    │  └─────────┘  └─────────┘  └─────────┘     │
    │         Fallback Chain: → → →                │
    └─────────────────────────────────────────────┘
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._providers: dict[str, ILLMProvider] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._fallback_chain: list[LLMProvider] = settings.llm_fallback_chain
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """
        Initialize configured providers.

        Only LOCAL (vLLM) is required. OpenAI is optional for research baselines.
        Gemini and Ollama have been removed from ChainMind.
        """
        settings = self._settings

        # ── Local vLLM (primary — the only required provider) ──────────────────
        if settings.local_vllm_enabled:
            try:
                from chainmind.llm.local_provider import LocalProvider
                local_provider = LocalProvider(
                    model_id=settings.local_vllm_model,
                    base_url=settings.local_vllm_base_url,
                    served_model_name=settings.local_vllm_served_name,
                )
                # Startup probe — verify vLLM is reachable
                import urllib.request
                probe_url = (
                    settings.local_vllm_base_url.rstrip("/").rstrip("v1").rstrip("/")
                    + "/v1/models"
                )
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(probe_url, method="GET"), timeout=5
                    ) as resp:
                        if resp.status == 200:
                            self._providers["local"] = local_provider
                            self._circuit_breakers["local"] = CircuitBreaker(
                                name="local_vllm",
                                failure_threshold=settings.llm_circuit_breaker_threshold,
                                recovery_timeout_seconds=settings.llm_circuit_breaker_recovery_seconds,
                            )
                            logger.info(
                                f"✓ Local vLLM ready: {settings.local_vllm_model} "
                                f"@ {settings.local_vllm_base_url}"
                            )
                        else:
                            logger.error(
                                f"vLLM probe returned {resp.status} — "
                                f"server may not be fully ready yet."
                            )
                except (urllib.request.URLError, OSError, TimeoutError) as e:
                    logger.error(
                        f"\n"
                        f"  ✘ Local vLLM not reachable at {settings.local_vllm_base_url}\n"
                        f"  Start it with: bash scripts/start_model_server.sh qwen2.5-7b\n"
                        f"  Error: {e}"
                    )
                    # Do NOT silently fall through — raise immediately
                    raise RuntimeError(
                        f"vLLM server not running at {settings.local_vllm_base_url}. "
                        f"Start it with: bash scripts/start_model_server.sh qwen2.5-7b"
                    )
            except RuntimeError:
                raise
            except ImportError:
                logger.error("LocalProvider import failed — openai package missing")
                raise

        # ── OpenAI (optional — GPT-4 research baseline ONLY) ──────────────────
        if settings.openai_api_keys:
            from chainmind.llm.openai_provider import OpenAIProvider
            self._providers["openai"] = OpenAIProvider(
                model=settings.openai_model,
                api_keys=settings.openai_api_keys,
            )
            self._circuit_breakers["openai"] = CircuitBreaker(
                name="openai",
                failure_threshold=settings.llm_circuit_breaker_threshold,
                recovery_timeout_seconds=settings.llm_circuit_breaker_recovery_seconds,
            )
            logger.info(
                f"OpenAI provider initialized (research baseline — {settings.openai_model})"
            )

        if not self._providers:
            raise RuntimeError(
                "No LLM providers available. "
                "Start vLLM: bash scripts/start_model_server.sh qwen2.5-7b"
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Route a generation request through the fallback chain.

        Tries each provider in order:
        1. Check circuit breaker state
        2. Attempt generation with key rotation
        3. On failure, move to next provider in chain
        """
        errors: list[str] = []

        for provider_name in self._fallback_chain:
            provider_key = provider_name.value

            # Skip unconfigured providers
            if provider_key not in self._providers:
                continue

            provider = self._providers[provider_key]
            circuit_breaker = self._circuit_breakers[provider_key]

            try:
                # Execute through circuit breaker
                response = await circuit_breaker.call(provider.generate, request)
                logger.debug(
                    f"LLM request served by {provider_key} "
                    f"({response.latency_ms:.0f}ms, {response.usage.total_tokens} tokens)"
                )
                return response

            except LLMCircuitOpenError:
                msg = f"{provider_key}: circuit breaker OPEN"
                logger.warning(msg)
                errors.append(msg)
                continue

            except LLMQuotaExhaustedError as e:
                msg = f"{provider_key}: all keys exhausted"
                logger.warning(msg)
                errors.append(msg)
                continue

            except LLMProviderError as e:
                msg = f"{provider_key}: {e}"
                logger.warning(f"Provider error, falling back: {msg}")
                errors.append(msg)
                continue

            except Exception as e:
                msg = f"{provider_key}: unexpected error: {e}"
                logger.error(msg)
                errors.append(msg)
                continue

        raise LLMAllProvidersFailedError(errors)

    async def generate_structured(
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> LLMResponse:
        """Route a structured generation request through the fallback chain."""
        errors: list[str] = []

        for provider_name in self._fallback_chain:
            provider_key = provider_name.value

            if provider_key not in self._providers:
                continue

            provider = self._providers[provider_key]
            circuit_breaker = self._circuit_breakers[provider_key]

            try:
                response = await circuit_breaker.call(
                    provider.generate_structured, request, schema
                )
                return response

            except (LLMCircuitOpenError, LLMQuotaExhaustedError, LLMProviderError) as e:
                errors.append(f"{provider_key}: {e}")
                continue

            except Exception as e:
                errors.append(f"{provider_key}: {e}")
                continue

        raise LLMAllProvidersFailedError(errors)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream from the first available provider."""
        for provider_name in self._fallback_chain:
            provider_key = provider_name.value

            if provider_key not in self._providers:
                continue

            cb = self._circuit_breakers[provider_key]
            if cb.state == CircuitState.OPEN:
                continue

            try:
                provider = self._providers[provider_key]
                async for token in provider.stream(request):
                    yield token
                return
            except Exception as e:
                logger.warning(f"Stream failed on {provider_key}: {e}, trying next...")
                continue

        raise LLMAllProvidersFailedError(["All providers failed for streaming"])

    async def health_check_all(self) -> dict[str, dict]:
        """Check health of all providers and their circuit breakers."""
        results = {}
        for name, provider in self._providers.items():
            cb = self._circuit_breakers[name]
            try:
                healthy = await provider.health_check()
            except Exception:
                healthy = False

            results[name] = {
                "healthy": healthy,
                "circuit_breaker": cb.to_dict(),
            }
        return results

    def get_provider(self, name: str) -> ILLMProvider | None:
        """Get a specific provider by name."""
        return self._providers.get(name)

    @property
    def available_providers(self) -> list[str]:
        """List of configured provider names."""
        return list(self._providers.keys())
