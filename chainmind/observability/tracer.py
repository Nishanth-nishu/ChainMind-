"""
ChainMind Tracer — OpenTelemetry-compatible distributed tracing.

Tracks agent reasoning steps, tool calls, LLM latencies,
and retrieval operations for end-to-end observability.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from chainmind.core.types import TraceSpan

logger = logging.getLogger(__name__)


class Tracer:
    """
    Distributed tracer for agentic workflows.

    Creates hierarchical spans that can be exported to
    OpenTelemetry, Jaeger, or simple JSON logs.
    """

    def __init__(self, service_name: str = "chainmind", enabled: bool = True):
        self._service_name = service_name
        self._enabled = enabled
        self._traces: dict[str, list[TraceSpan]] = {}

    @asynccontextmanager
    async def span(
        self,
        operation: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncGenerator[TraceSpan, None]:
        """
        Create a trace span as an async context manager.

        Usage:
            async with tracer.span("llm_generate", trace_id="abc") as span:
                result = await llm.generate(...)
                span.attributes["tokens"] = result.usage.total_tokens
        """
        if not self._enabled:
            # No-op span
            yield TraceSpan(
                trace_id=trace_id or str(uuid.uuid4()),
                operation=operation,
            )
            return

        tid = trace_id or str(uuid.uuid4())
        span = TraceSpan(
            trace_id=tid,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=datetime.now(timezone.utc),
            attributes={
                "service": self._service_name,
                **(attributes or {}),
            },
        )

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            span.attributes["error_type"] = type(e).__name__
            raise
        finally:
            span.end_time = datetime.now(timezone.utc)
            duration = (span.end_time - span.start_time).total_seconds() * 1000
            span.attributes["duration_ms"] = round(duration, 2)

            # Store span
            if tid not in self._traces:
                self._traces[tid] = []
            self._traces[tid].append(span)

            logger.debug(
                f"Span completed: {operation} ({duration:.1f}ms) "
                f"[{span.status}]",
                extra={"trace_id": tid, "span_id": span.span_id},
            )

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """Get all spans for a trace ID."""
        return self._traces.get(trace_id, [])

    def get_all_traces(self) -> dict[str, list[TraceSpan]]:
        """Get all stored traces."""
        return dict(self._traces)

    def export_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Export a trace as a list of dicts (OTLP-compatible)."""
        spans = self.get_trace(trace_id)
        return [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "operation": s.operation,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in spans
        ]

    def clear(self) -> None:
        """Clear all stored traces."""
        self._traces.clear()


# Global tracer instance
_global_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get or create the global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def init_tracer(service_name: str = "chainmind", enabled: bool = True) -> Tracer:
    """Initialize the global tracer."""
    global _global_tracer
    _global_tracer = Tracer(service_name=service_name, enabled=enabled)
    return _global_tracer
