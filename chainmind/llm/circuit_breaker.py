"""
ChainMind Circuit Breaker — Self-healing pattern for LLM providers.

States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (probing recovery)

Based on the Hystrix/Resilience4j circuit breaker pattern adapted for
non-deterministic AI systems. When a provider consistently fails, the
circuit opens to prevent cascading failures and wasted quota.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from chainmind.config.constants import CIRCUIT_BREAKER_MIN_CALLS, CircuitState
from chainmind.core.exceptions import LLMCircuitOpenError

logger = logging.getLogger(__name__)


@dataclass
class CircuitMetrics:
    """Sliding window metrics for circuit breaker decisions."""
    total_calls: int = 0
    failed_calls: int = 0
    successful_calls: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for LLM provider self-healing.

    - CLOSED: Normal operation. Tracks failure rate.
    - OPEN: Provider is failing. All calls rejected immediately.
                Transitions to HALF_OPEN after recovery_timeout.
    - HALF_OPEN: Allows a single probe call. If it succeeds,
                 transitions to CLOSED. If it fails, back to OPEN.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        success_threshold: int = 2,
    ):
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._success_threshold = success_threshold
        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> CircuitState:
        """Current state, with automatic OPEN→HALF_OPEN transition on timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit '{self._name}' transitioning OPEN → HALF_OPEN "
                    f"after {elapsed:.1f}s recovery period"
                )
        return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        return self._metrics

    async def call(self, func, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Raises LLMCircuitOpenError if the circuit is OPEN.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                raise LLMCircuitOpenError(self._name)

        # Execute the actual call
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle a successful call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                if self._metrics.consecutive_successes >= self._success_threshold:
                    self._state = CircuitState.CLOSED
                    self._metrics.consecutive_failures = 0
                    logger.info(
                        f"Circuit '{self._name}' recovered: HALF_OPEN → CLOSED "
                        f"after {self._success_threshold} successful probes"
                    )

    async def _on_failure(self) -> None:
        """Handle a failed call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — back to OPEN
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    f"Circuit '{self._name}' probe failed: HALF_OPEN → OPEN"
                )

            elif self._state == CircuitState.CLOSED:
                if (
                    self._metrics.total_calls >= CIRCUIT_BREAKER_MIN_CALLS
                    and self._metrics.consecutive_failures >= self._failure_threshold
                ):
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    logger.warning(
                        f"Circuit '{self._name}' tripped: CLOSED → OPEN "
                        f"({self._metrics.consecutive_failures} consecutive failures)"
                    )

    async def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._metrics = CircuitMetrics()
            logger.info(f"Circuit '{self._name}' manually reset to CLOSED")

    def to_dict(self) -> dict:
        """Serialize state for observability."""
        return {
            "name": self._name,
            "state": self._state.value,
            "metrics": {
                "total_calls": self._metrics.total_calls,
                "failed_calls": self._metrics.failed_calls,
                "successful_calls": self._metrics.successful_calls,
                "consecutive_failures": self._metrics.consecutive_failures,
                "failure_rate": (
                    self._metrics.failed_calls / max(self._metrics.total_calls, 1)
                ),
            },
        }
