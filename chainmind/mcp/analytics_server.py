"""
ChainMind Analytics MCP Server — Computation and analysis tools.

Provides analytical tools: reorder point calculation,
lead time analysis, anomaly detection.
"""

from __future__ import annotations

import math
import random
from typing import Any

from chainmind.config.constants import ToolCategory
from chainmind.core.types import MCPToolDefinition
from chainmind.mcp.base_server import BaseMCPServer


class AnalyticsMCPServer(BaseMCPServer):
    """MCP server for supply chain analytics computations."""

    def __init__(self):
        super().__init__(name="analytics")
        self._register_all_tools()

    def _register_all_tools(self) -> None:
        self._register_tool(
            MCPToolDefinition(
                name="calculate_reorder_point",
                description="Calculate the optimal reorder point for a SKU based on demand and lead time",
                category=ToolCategory.ANALYTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "sku_id": {"type": "string", "description": "SKU identifier"},
                        "service_level": {"type": "number", "description": "Target service level (0.0-1.0, default: 0.95)"},
                    },
                },
                required_params=["sku_id"],
            ),
            self._calculate_reorder_point,
        )

        self._register_tool(
            MCPToolDefinition(
                name="analyze_lead_times",
                description="Analyze supplier lead time performance and variability",
                category=ToolCategory.ANALYTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "supplier_id": {"type": "string", "description": "Supplier identifier"},
                    },
                },
                required_params=["supplier_id"],
            ),
            self._analyze_lead_times,
        )

        self._register_tool(
            MCPToolDefinition(
                name="detect_anomalies",
                description="Detect anomalies in supply chain metrics using statistical methods",
                category=ToolCategory.ANALYTICS,
                parameters={
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string", "description": "Metric to analyze (e.g., demand, quality, lead_time)"},
                        "window_days": {"type": "integer", "description": "Analysis window in days (default: 30)"},
                    },
                },
                required_params=["metric_name"],
            ),
            self._detect_anomalies,
        )

    async def _calculate_reorder_point(
        self, sku_id: str, service_level: float = 0.95
    ) -> dict[str, Any]:
        """Calculate reorder point using (avg_demand * lead_time) + safety_stock."""
        # Simulated demand statistics
        avg_daily_demand = random.uniform(20, 150)
        demand_std = avg_daily_demand * random.uniform(0.1, 0.3)
        avg_lead_time = random.uniform(7, 21)
        lead_time_std = avg_lead_time * random.uniform(0.1, 0.2)

        # Z-score for service level
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z = z_scores.get(round(service_level, 2), 1.65)

        # Safety stock = Z * sqrt(LT * σd² + d² * σLT²)
        safety_stock = z * math.sqrt(
            avg_lead_time * demand_std**2 + avg_daily_demand**2 * lead_time_std**2
        )

        # Reorder point = avg_demand * avg_lead_time + safety_stock
        rop = avg_daily_demand * avg_lead_time + safety_stock

        return {
            "sku_id": sku_id,
            "reorder_point": round(rop),
            "safety_stock": round(safety_stock),
            "avg_daily_demand": round(avg_daily_demand, 1),
            "avg_lead_time_days": round(avg_lead_time, 1),
            "service_level": service_level,
            "recommendation": (
                f"Set reorder point to {round(rop)} units with safety stock of {round(safety_stock)} units "
                f"to achieve {service_level*100}% service level"
            ),
        }

    async def _analyze_lead_times(self, supplier_id: str) -> dict[str, Any]:
        """Analyze lead time performance."""
        # Simulated lead time history
        avg_lt = random.uniform(10, 25)
        samples = [avg_lt + random.gauss(0, avg_lt * 0.15) for _ in range(50)]
        actual_avg = sum(samples) / len(samples)
        std_dev = (sum((x - actual_avg)**2 for x in samples) / len(samples)) ** 0.5
        on_time_pct = sum(1 for x in samples if x <= avg_lt * 1.1) / len(samples)

        return {
            "supplier_id": supplier_id,
            "avg_lead_time_days": round(actual_avg, 1),
            "std_deviation_days": round(std_dev, 1),
            "min_lead_time": round(min(samples), 1),
            "max_lead_time": round(max(samples), 1),
            "on_time_delivery_pct": round(on_time_pct * 100, 1),
            "reliability_score": round(on_time_pct, 2),
            "trend": "stable",
            "sample_size": len(samples),
        }

    async def _detect_anomalies(
        self, metric_name: str, window_days: int = 30
    ) -> dict[str, Any]:
        """Detect anomalies using z-score method."""
        # Simulated metric data
        mean = random.uniform(50, 200)
        std = mean * 0.15
        data_points = [mean + random.gauss(0, std) for _ in range(window_days)]

        # Inject anomalies
        anomaly_idx = random.sample(range(window_days), k=min(3, window_days))
        for idx in anomaly_idx:
            data_points[idx] = mean + random.choice([-1, 1]) * std * random.uniform(2.5, 4.0)

        # Detect via z-score
        actual_mean = sum(data_points) / len(data_points)
        actual_std = (sum((x - actual_mean)**2 for x in data_points) / len(data_points)) ** 0.5
        threshold = 2.0

        anomalies = []
        for i, val in enumerate(data_points):
            z_score = abs(val - actual_mean) / max(actual_std, 0.001)
            if z_score > threshold:
                anomalies.append({
                    "day": i + 1,
                    "value": round(val, 2),
                    "z_score": round(z_score, 2),
                    "severity": "high" if z_score > 3 else "medium",
                })

        return {
            "metric_name": metric_name,
            "window_days": window_days,
            "mean": round(actual_mean, 2),
            "std_deviation": round(actual_std, 2),
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "health_status": "warning" if anomalies else "healthy",
        }
