"""
ChainMind Metrics — Performance metrics collector.

Tracks agent success/failure rates, LLM latencies,
retrieval quality, and circuit breaker states.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricPoint:
    """A single metric observation."""
    value: float
    timestamp: float = field(default_factory=time.monotonic)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    In-memory metrics collector with Prometheus-compatible export.

    Tracks counters, gauges, and histograms.
    """

    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        # Cap history to prevent unbounded growth
        if len(self._histograms[key]) > 10000:
            self._histograms[key] = self._histograms[key][-5000:]

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0.0)

    def get_histogram_stats(self, name: str, labels: dict[str, str] | None = None) -> dict[str, float]:
        """Get histogram statistics (count, mean, p50, p95, p99)."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "count": n,
            "mean": sum(sorted_vals) / n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "p50": sorted_vals[int(n * 0.5)],
            "p95": sorted_vals[int(n * 0.95)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for key, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {key.split('{')[0]} counter")
            lines.append(f"{key} {value}")

        for key, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {key.split('{')[0]} gauge")
            lines.append(f"{key} {value}")

        for key in sorted(self._histograms.keys()):
            stats = self.get_histogram_stats(key)
            base_name = key.split("{")[0]
            labels = key[len(base_name):]
            lines.append(f"# TYPE {base_name} histogram")
            lines.append(f'{base_name}_count{labels} {stats["count"]}')
            lines.append(f'{base_name}_mean{labels} {stats["mean"]:.2f}')
            lines.append(f'{base_name}_p95{labels} {stats["p95"]:.2f}')
            lines.append(f'{base_name}_p99{labels} {stats["p99"]:.2f}')

        return "\n".join(lines)

    def export_dict(self) -> dict[str, Any]:
        """Export all metrics as a dict."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: self.get_histogram_stats(k) for k in self._histograms
            },
        }

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics instance
_global_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
