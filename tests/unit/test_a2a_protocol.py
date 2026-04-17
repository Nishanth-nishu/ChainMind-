"""
Unit tests for A2A Protocol — Agent registry, discovery, and routing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from chainmind.a2a.protocol import AgentRegistry, A2ABus
from chainmind.config.constants import AgentRole, TaskStatus
from chainmind.core.types import AgentCard, AgentContext, TaskRequest, TaskResponse


def _make_mock_agent(role: AgentRole, capabilities: list[str]) -> MagicMock:
    """Create a mock agent with the given role and capabilities."""
    agent = MagicMock()
    agent.agent_card = AgentCard(
        name=f"Test {role.value}",
        role=role,
        description=f"Test agent for {role.value}",
        capabilities=capabilities,
    )
    agent.process = AsyncMock(return_value=TaskResponse(
        task_id="test-task",
        source_agent=agent.agent_card.agent_id,
        status=TaskStatus.COMPLETED,
        result=f"Result from {role.value}",
    ))
    return agent


@pytest.mark.unit
class TestAgentRegistry:

    def test_register_agent(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.INVENTORY_MANAGEMENT, ["stock_monitoring"])
        registry.register(agent)

        assert len(registry.list_all()) == 1

    def test_discover_by_role(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.DEMAND_FORECASTING, ["demand_prediction"])
        registry.register(agent)

        found = registry.discover_by_role(AgentRole.DEMAND_FORECASTING)
        assert len(found) == 1
        assert found[0].role == AgentRole.DEMAND_FORECASTING

    def test_discover_by_capability(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.LOGISTICS, ["route_optimization", "shipment_tracking"])
        registry.register(agent)

        found = registry.discover_by_capability("route_optimization")
        assert len(found) == 1

    def test_deregister_agent(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.PROCUREMENT, ["supplier_evaluation"])
        registry.register(agent)
        assert len(registry.list_all()) == 1

        registry.deregister(agent.agent_card.agent_id)
        assert len(registry.list_all()) == 0

    @pytest.mark.asyncio
    async def test_route_task_by_role(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.INVENTORY_MANAGEMENT, ["stock_monitoring"])
        registry.register(agent)

        task = TaskRequest(
            source_agent="test",
            target_agent="inventory_management",
            query="Check stock levels",
        )
        context = AgentContext()

        response = await registry.route_task(task, context)
        assert response.status == TaskStatus.COMPLETED
        agent.process.assert_called_once()


@pytest.mark.unit
class TestA2ABus:

    @pytest.mark.asyncio
    async def test_send_task(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.QUALITY_ASSURANCE, ["anomaly_detection"])
        registry.register(agent)

        bus = A2ABus(registry)

        task = TaskRequest(
            source_agent="orchestrator",
            target_agent="quality_assurance",
            query="Detect anomalies",
        )
        context = AgentContext()

        response = await bus.send_task(task, context)
        assert response.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_log(self):
        registry = AgentRegistry()
        agent = _make_mock_agent(AgentRole.DEMAND_FORECASTING, ["demand_prediction"])
        registry.register(agent)

        bus = A2ABus(registry)

        task = TaskRequest(
            source_agent="orchestrator",
            target_agent="demand_forecasting",
            query="Forecast demand",
        )
        context = AgentContext()

        response = await bus.send_task(task, context)
        logged = bus.get_task_status(task.task_id)
        assert logged is not None
        assert logged.status == TaskStatus.COMPLETED
