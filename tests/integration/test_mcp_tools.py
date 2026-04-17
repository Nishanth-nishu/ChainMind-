"""
Integration tests for the MCP tool execution pipeline.

Tests the full flow: Agent → MCP Server → Tool Handler → Response
"""

from chainmind.core.exceptions import MCPToolNotFoundError
import pytest
from chainmind.mcp.supply_chain_server import SupplyChainMCPServer
from chainmind.mcp.analytics_server import AnalyticsMCPServer
from chainmind.mcp.knowledge_base_server import KnowledgeBaseMCPServer
from chainmind.retrieval.knowledge_base import KnowledgeBase
from chainmind.config.settings import get_settings


@pytest.mark.integration
class TestSupplyChainMCPServer:

    @pytest.fixture
    def server(self):
        return SupplyChainMCPServer()

    @pytest.mark.asyncio
    async def test_get_inventory_levels(self, server):
        result = await server.execute_tool(
            "get_inventory_levels",
            {"sku_id": "SKU-001"},
        )
        assert result.success
        assert result.result is not None
        data = result.result
        assert "sku_id" in data
        assert "quantity" in data

    @pytest.mark.asyncio
    async def test_get_demand_forecast(self, server):
        result = await server.execute_tool(
            "get_demand_forecast",
            {"product_id": "WIDGET-A", "horizon_days": 30},
        )
        assert result.success
        assert "forecast" in result.result

    @pytest.mark.asyncio
    async def test_get_supplier_info(self, server):
        result = await server.execute_tool(
            "get_supplier_info",
            {"supplier_id": "SUP-001"},
        )
        assert result.success
        assert "supplier_id" in result.result

    @pytest.mark.asyncio
    async def test_get_order_status(self, server):
        result = await server.execute_tool(
            "get_order_status",
            {"order_id": "ORD-1001"},
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_get_shipment_tracking(self, server):
        result = await server.execute_tool(
            "get_shipment_tracking",
            {"shipment_id": "SHIP-2001"},
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self, server):
        with pytest.raises(MCPToolNotFoundError):
            await server.execute_tool(
                "nonexistent_tool",
                {},
            )

    @pytest.mark.asyncio
    async def test_missing_optional_param_uses_default(self, server):
        """Simulated handlers have defaults — missing params return all data."""
        result = await server.execute_tool(
            "get_inventory_levels",
            {},  # sku_id is optional in simulated handler
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        tools = server.list_tools()
        assert len(tools) >= 5
        tool_names = [t.name for t in tools]
        assert "get_inventory_levels" in tool_names
        assert "get_demand_forecast" in tool_names


@pytest.mark.integration
class TestAnalyticsMCPServer:

    @pytest.fixture
    def server(self):
        return AnalyticsMCPServer()

    @pytest.mark.asyncio
    async def test_calculate_reorder_point(self, server):
        result = await server.execute_tool(
            "calculate_reorder_point",
            {"sku_id": "SKU-001", "service_level": 0.95},
        )
        assert result.success
        data = result.result
        assert "reorder_point" in data
        assert "safety_stock" in data

    @pytest.mark.asyncio
    async def test_analyze_lead_times(self, server):
        result = await server.execute_tool(
            "analyze_lead_times",
            {"supplier_id": "SUP-001"},
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_detect_anomalies(self, server):
        result = await server.execute_tool(
            "detect_anomalies",
            {"metric_name": "demand", "window_days": 30},
        )
        assert result.success
        assert "anomalies" in result.result


@pytest.mark.integration
class TestKnowledgeBaseMCPServer:

    @pytest.fixture
    def server(self):
        settings = get_settings()
        kb = KnowledgeBase(settings)
        return KnowledgeBaseMCPServer(knowledge_base=kb)

    @pytest.mark.asyncio
    async def test_search_knowledge_base(self, server):
        result = await server.execute_tool(
            "search_knowledge_base",
            {"query": "safety stock", "top_k": 3},
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_ingest_and_retrieve(self, server):
        # Ingest — use title/category params matching handler signature
        ingest_result = await server.execute_tool(
            "ingest_document",
            {
                "content": "Test document about warehouse management best practices for inventory rotation.",
                "title": "Test Doc",
                "category": "test",
            },
        )
        assert ingest_result.success

        # Search
        search_result = await server.execute_tool(
            "search_knowledge_base",
            {"query": "warehouse inventory rotation", "top_k": 3},
        )
        assert search_result.success
