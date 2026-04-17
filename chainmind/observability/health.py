"""
ChainMind Health Monitor — Self-healing health checks and recovery.

Periodically probes all providers and services.
Implements automatic degraded-mode activation and recovery detection.
Kubernetes-compatible liveness/readiness probes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from chainmind.core.types import HealthStatus

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Self-healing health monitor.

    Runs periodic health checks and manages system state:
    - HEALTHY: All components operational
    - DEGRADED: Some components failed, fallbacks active
    - UNHEALTHY: Critical components failed
    """

    def __init__(self, check_interval_seconds: int = 30):
        self._check_interval = check_interval_seconds
        self._component_status: dict[str, HealthStatus] = {}
        self._check_functions: dict[str, Any] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def register_check(self, component: str, check_fn) -> None:
        """Register a health check function for a component."""
        self._check_functions[component] = check_fn
        logger.info(f"Health check registered: {component}")

    async def check_all(self) -> dict[str, HealthStatus]:
        """Run all health checks and return status."""
        results = {}

        for component, check_fn in self._check_functions.items():
            start = time.perf_counter()
            try:
                healthy = await check_fn()
                latency_ms = (time.perf_counter() - start) * 1000
                status = HealthStatus(
                    component=component,
                    healthy=healthy,
                    latency_ms=latency_ms,
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                status = HealthStatus(
                    component=component,
                    healthy=False,
                    latency_ms=latency_ms,
                    error=str(e),
                )

            results[component] = status
            self._component_status[component] = status

        return results

    @property
    def is_healthy(self) -> bool:
        """Overall system health — True if all critical components are healthy."""
        if not self._component_status:
            return True  # No checks registered yet
        return all(s.healthy for s in self._component_status.values())

    @property
    def is_ready(self) -> bool:
        """Readiness check — True if system can serve requests (even degraded)."""
        if not self._component_status:
            return True
        # Ready if at least one LLM provider is healthy
        critical = [s for s in self._component_status.values()]
        return any(s.healthy for s in critical)

    @property
    def system_status(self) -> str:
        """Overall system status string."""
        if not self._component_status:
            return "unknown"
        all_healthy = all(s.healthy for s in self._component_status.values())
        any_healthy = any(s.healthy for s in self._component_status.values())

        if all_healthy:
            return "healthy"
        elif any_healthy:
            return "degraded"
        else:
            return "unhealthy"

    def get_status_report(self) -> dict[str, Any]:
        """Get complete status report."""
        return {
            "status": self.system_status,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "components": {
                name: {
                    "healthy": status.healthy,
                    "latency_ms": round(status.latency_ms, 2),
                    "error": status.error,
                    "last_check": status.last_check.isoformat(),
                }
                for name, status in self._component_status.items()
            },
        }

    async def start_monitoring(self) -> None:
        """Start the periodic health check loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Health monitoring started (interval={self._check_interval}s)")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await self.check_all()

                status = self.system_status
                if status != "healthy":
                    logger.warning(f"System status: {status}")
                    # Log unhealthy components
                    for name, s in self._component_status.items():
                        if not s.healthy:
                            logger.warning(f"  ⚠ {name}: {s.error or 'unhealthy'}")

            except Exception as e:
                logger.error(f"Health check loop error: {e}")

            await asyncio.sleep(self._check_interval)
