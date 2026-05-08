"""
chainmind/llm/circuit_breaker.py
Circuit breaker for LLM provider fault isolation.

State machine:  CLOSED → (failures ≥ threshold) → OPEN → (recovery timeout) → HALF_OPEN → CLOSED
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine

from chainmind.config.constants import CircuitState

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Async circuit breaker for any awaitable operation.

    Usage:
        cb = CircuitBreaker(name="openai", failure_threshold=3, recovery_timeout_seconds=60)
        result = await cb.call(provider.generate, request)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 60.0,
    ):
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._success_count = 0  # in HALF_OPEN state

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may auto-transition OPEN→HALF_OPEN on timeout)."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and (time.monotonic() - self._last_failure_time) >= self._recovery_timeout
        ):
            logger.info(f"Circuit '{self._name}': OPEN → HALF_OPEN (recovery window elapsed)")
            self._state = CircuitState.HALF_OPEN
            self._success_count = 0
        return self._state

    async def call(self, fn: Callable[..., Coroutine], *args: Any, **kwargs: Any) -> Any:
        """
        Execute fn(*args, **kwargs) through the circuit breaker.

        Raises:
            LLMCircuitOpenError: if the circuit is OPEN.
        """
        from chainmind.core.exceptions import LLMCircuitOpenError

        current = self.state
        if current == CircuitState.OPEN:
            raise LLMCircuitOpenError(
                f"Circuit '{self._name}' is OPEN. Try again after "
                f"{self._recovery_timeout}s."
            )

        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= 1:
                logger.info(f"Circuit '{self._name}': HALF_OPEN → CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        logger.warning(
            f"Circuit '{self._name}': failure {self._failure_count}/{self._failure_threshold}"
        )
        if self._failure_count >= self._failure_threshold:
            logger.error(f"Circuit '{self._name}': CLOSED → OPEN (threshold exceeded)")
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    def to_dict(self) -> dict:
        return {
            "name": self._name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_s": self._recovery_timeout,
        }
