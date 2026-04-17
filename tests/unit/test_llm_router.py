"""
Unit tests for the LLM Router — round-robin, failover, and circuit breaker integration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chainmind.llm.router import LLMRouter
from chainmind.config.settings import Settings, LLMProvider
from chainmind.core.types import LLMRequest, LLMResponse, LLMMessage, TokenUsage
from chainmind.core.exceptions import LLMAllProvidersFailedError


def _make_settings(**overrides) -> Settings:
    """Create test settings."""
    defaults = {
        "gemini_api_keys": ["test-key-1", "test-key-2"],
        "gemini_model": "gemini-2.0-flash",
        "openai_api_keys": [],
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama3.2",
        "llm_fallback_chain": [LLMProvider.GEMINI, LLMProvider.OLLAMA],
        "llm_circuit_breaker_threshold": 3,
        "llm_circuit_breaker_recovery_seconds": 1,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_response(content: str = "test response") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test",
        provider="test",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        latency_ms=100,
    )


@pytest.mark.unit
class TestLLMRouter:

    @pytest.mark.asyncio
    async def test_available_providers(self):
        settings = _make_settings()
        with patch("chainmind.llm.router.LLMRouter._initialize_providers"):
            router = LLMRouter(settings)
            # Router initializes but we mock it — test structure only
            assert router._settings == settings

    @pytest.mark.asyncio
    async def test_health_check_all_structure(self):
        """Verify health check returns expected structure."""
        settings = _make_settings()
        with patch("chainmind.llm.router.LLMRouter._initialize_providers"):
            router = LLMRouter(settings)
            router._providers = {}
            router._circuit_breakers = {}

            result = await router.health_check_all()
            assert isinstance(result, dict)
