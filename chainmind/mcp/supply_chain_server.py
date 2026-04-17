"""
ChainMind Supply Chain MCP Server — Domain-specific tools for supply chain operations.

Provides tools for inventory, demand, supplier, and order management.
Uses simulated data for demonstration — replace handlers with real
ERP/WMS integrations in production.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from chainmind.config.constants import ToolCategory
from chainmind.core.types import MCPToolDefinition
from chainmind.mcp.base_server import BaseMCPServer


# Simulated supply chain data (replace with real DB/API calls in production)
_INVENTORY_DATA = {
    "SKU-001": {"name": "Widget A", "warehouse": "WH-EAST", "quantity": 1250, "reorder_point": 500, "unit_cost": 12.50},
    "SKU-002": {"name": "Widget B", "warehouse": "WH-WEST", "quantity": 340, "reorder_point": 400, "unit_cost": 28.75},
    "SKU-003": {"name": "Component X", "warehouse": "WH-CENTRAL", "quantity": 5600, "reorder_point": 2000, "unit_cost": 3.20},
    "SKU-004": {"name": "Assembly Kit Y", "warehouse": "WH-EAST", "quantity": 89, "reorder_point": 200, "unit_cost": 145.00},
    "SKU-005": {"name": "Raw Material Z", "warehouse": "WH-SOUTH", "quantity": 15000, "reorder_point": 5000, "unit_cost": 0.85},
}

_SUPPLIER_DATA = {
    "SUP-001": {"name": "Acme Components", "rating": 4.2, "lead_time_days": 14, "reliability": 0.95, "location": "Shanghai, CN"},
    "SUP-002": {"name": "GlobalParts Inc", "rating": 3.8, "lead_time_days": 21, "reliability": 0.88, "location": "Detroit, US"},
    "SUP-003": {"name": "EuroPrecision GmbH", "rating": 4.7, "lead_time_days": 18, "reliability": 0.97, "location": "Munich, DE"},
}

_ORDER_DATA = {
    "ORD-1001": {"status": "shipped", "items": ["SKU-001", "SKU-003"], "carrier": "FedEx", "eta": "2025-12-20"},
    "ORD-1002": {"status": "processing", "items": ["SKU-004"], "carrier": "DHL", "eta": "2025-12-28"},
    "ORD-1003": {"status": "delivered", "items": ["SKU-002", "SKU-005"], "carrier": "UPS", "delivered_date": "2025-12-10"},
}


class SupplyChainMCPServer(BaseMCPServer):
    """MCP server exposing supply chain data tools."""

    def __init__(self):
        super().__init__(name="supply_chain")
        self._register_all_tools()

    def _register_all_tools(self) -> None:
        """Register all supply chain tools."""

        self._register_tool(
            MCPToolDefinition(
                name="get_inventory_levels",
                description="Get current inventory levels for a specific SKU or all SKUs in a warehouse",
                category=ToolCategory.SUPPLY_CHAIN,
                parameters={
                    "type": "object",
                    "properties": {
                        "sku_id": {"type": "string", "description": "SKU identifier (e.g., SKU-001). Leave empty for all."},
                        "warehouse_id": {"type": "string", "description": "Warehouse ID filter (e.g., WH-EAST). Optional."},
                    },
                },
            ),
            self._get_inventory_levels,
        )

        self._register_tool(
            MCPToolDefinition(
                name="get_demand_forecast",
                description="Get demand forecast for a product over a specified horizon",
                category=ToolCategory.SUPPLY_CHAIN,
                parameters={
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "Product/SKU identifier"},
                        "horizon_days": {"type": "integer", "description": "Forecast horizon in days (default: 30)"},
                    },
                },
                required_params=["product_id"],
            ),
            self._get_demand_forecast,
        )

        self._register_tool(
            MCPToolDefinition(
                name="get_supplier_info",
                description="Get supplier details including rating, lead time, and reliability metrics",
                category=ToolCategory.SUPPLY_CHAIN,
                parameters={
                    "type": "object",
                    "properties": {
                        "supplier_id": {"type": "string", "description": "Supplier identifier (e.g., SUP-001)"},
                    },
                },
                required_params=["supplier_id"],
            ),
            self._get_supplier_info,
        )

        self._register_tool(
            MCPToolDefinition(
                name="get_order_status",
                description="Get current status and details of a purchase/sales order",
                category=ToolCategory.SUPPLY_CHAIN,
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order identifier (e.g., ORD-1001)"},
                    },
                },
                required_params=["order_id"],
            ),
            self._get_order_status,
        )

        self._register_tool(
            MCPToolDefinition(
                name="get_shipment_tracking",
                description="Track a shipment by order ID and get current location and ETA",
                category=ToolCategory.SUPPLY_CHAIN,
                parameters={
                    "type": "object",
                    "properties": {
                        "shipment_id": {"type": "string", "description": "Shipment/Order ID to track"},
                    },
                },
                required_params=["shipment_id"],
            ),
            self._get_shipment_tracking,
        )

    # === Tool Handlers ===

    async def _get_inventory_levels(
        self, sku_id: str = "", warehouse_id: str = ""
    ) -> dict[str, Any]:
        """Get inventory levels from simulated data."""
        if sku_id and sku_id in _INVENTORY_DATA:
            data = _INVENTORY_DATA[sku_id]
            status = "below_reorder" if data["quantity"] < data["reorder_point"] else "healthy"
            return {
                "sku_id": sku_id,
                **data,
                "status": status,
                "days_of_supply": round(data["quantity"] / max(random.randint(10, 50), 1)),
            }

        # Return all or filtered by warehouse
        results = []
        for sid, data in _INVENTORY_DATA.items():
            if warehouse_id and data["warehouse"] != warehouse_id:
                continue
            status = "below_reorder" if data["quantity"] < data["reorder_point"] else "healthy"
            results.append({"sku_id": sid, **data, "status": status})

        return {"inventory": results, "total_skus": len(results)}

    async def _get_demand_forecast(
        self, product_id: str, horizon_days: int = 30
    ) -> dict[str, Any]:
        """Generate simulated demand forecast."""
        base_demand = random.randint(50, 200)
        forecast = []
        for day in range(horizon_days):
            date = datetime.now(timezone.utc) + timedelta(days=day)
            seasonal_factor = 1.0 + 0.2 * (1 if date.month in [11, 12] else 0)  # Holiday spike
            daily_demand = int(base_demand * seasonal_factor * random.uniform(0.8, 1.2))
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_demand": daily_demand,
                "confidence_low": int(daily_demand * 0.85),
                "confidence_high": int(daily_demand * 1.15),
            })

        return {
            "product_id": product_id,
            "horizon_days": horizon_days,
            "total_predicted": sum(f["predicted_demand"] for f in forecast),
            "avg_daily_demand": round(sum(f["predicted_demand"] for f in forecast) / horizon_days),
            "forecast": forecast[:7],  # Return first 7 days detail
            "trend": "increasing" if forecast[-1]["predicted_demand"] > forecast[0]["predicted_demand"] else "stable",
        }

    async def _get_supplier_info(self, supplier_id: str) -> dict[str, Any]:
        """Get supplier information."""
        if supplier_id not in _SUPPLIER_DATA:
            return {"error": f"Supplier '{supplier_id}' not found", "available": list(_SUPPLIER_DATA.keys())}
        return {"supplier_id": supplier_id, **_SUPPLIER_DATA[supplier_id]}

    async def _get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get order status."""
        if order_id not in _ORDER_DATA:
            return {"error": f"Order '{order_id}' not found", "available": list(_ORDER_DATA.keys())}
        return {"order_id": order_id, **_ORDER_DATA[order_id]}

    async def _get_shipment_tracking(self, shipment_id: str) -> dict[str, Any]:
        """Track a shipment."""
        order = _ORDER_DATA.get(shipment_id)
        if not order:
            return {"error": f"Shipment '{shipment_id}' not found"}

        tracking_events = [
            {"timestamp": "2025-12-10T08:00:00Z", "location": "Origin Warehouse", "status": "picked_up"},
            {"timestamp": "2025-12-11T14:30:00Z", "location": "Regional Hub", "status": "in_transit"},
            {"timestamp": "2025-12-12T09:00:00Z", "location": "Distribution Center", "status": "sorting"},
        ]

        if order["status"] == "delivered":
            tracking_events.append(
                {"timestamp": "2025-12-13T16:00:00Z", "location": "Destination", "status": "delivered"}
            )

        return {
            "shipment_id": shipment_id,
            "carrier": order.get("carrier", "Unknown"),
            "status": order["status"],
            "eta": order.get("eta", "N/A"),
            "tracking_events": tracking_events,
        }
