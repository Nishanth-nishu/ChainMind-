"""
ChainMind Rate Limiter — Token and request rate limiting.

Sliding window rate limiter for API requests and token budgets.
Prevents runaway costs from agent loops.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in a time window and enforces limits
    for both request count and token consumption.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 60,
        max_tokens_per_minute: int = 100000,
        window_seconds: int = 60,
    ):
        self._max_rpm = max_requests_per_minute
        self._max_tpm = max_tokens_per_minute
        self._window = window_seconds
        self._request_times: deque[float] = deque()
        self._token_log: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    async def check_and_consume(self, tokens: int = 0) -> bool:
        """
        Check if request is within rate limits and consume quota.

        Returns True if allowed, False if rate-limited.
        """
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self._window

            # Prune old entries
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            while self._token_log and self._token_log[0][0] < cutoff:
                self._token_log.popleft()

            # Check request rate
            if len(self._request_times) >= self._max_rpm:
                logger.warning(f"Rate limit exceeded: {len(self._request_times)} requests/min")
                return False

            # Check token rate
            current_tokens = sum(t[1] for t in self._token_log)
            if current_tokens + tokens > self._max_tpm:
                logger.warning(f"Token limit exceeded: {current_tokens + tokens}/{self._max_tpm}")
                return False

            # Consume
            self._request_times.append(now)
            if tokens > 0:
                self._token_log.append((now, tokens))

            return True

    async def wait_if_needed(self, tokens: int = 0) -> None:
        """Wait until rate limit allows the request."""
        while not await self.check_and_consume(tokens):
            await asyncio.sleep(1.0)

    @property
    def current_rpm(self) -> int:
        """Current requests in the window."""
        now = time.monotonic()
        cutoff = now - self._window
        return sum(1 for t in self._request_times if t >= cutoff)

    @property
    def current_tpm(self) -> int:
        """Current tokens in the window."""
        now = time.monotonic()
        cutoff = now - self._window
        return sum(t[1] for t in self._token_log if t[0] >= cutoff)
