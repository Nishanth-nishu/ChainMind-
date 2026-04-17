"""
ChainMind A2A Protocol — Agent-to-Agent communication protocol.

Implements the A2A specification:
- Agent discovery and capability advertisement
- Task lifecycle management (PENDING → IN_PROGRESS → COMPLETED/FAILED)
- Message envelope with correlation IDs
- Streaming status updates
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from chainmind.config.constants import AgentRole, TaskStatus
from chainmind.core.exceptions import AgentNotFoundError, TaskRoutingError
from chainmind.core.interfaces import IAgent
from chainmind.core.types import (
    AgentCard,
    AgentContext,
    TaskRequest,
    TaskResponse,
)

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    A2A Agent Registry — Discovery and capability-based routing.

    Maintains a registry of all agents and their capabilities.
    Routes tasks to the most appropriate agent based on:
    1. Target agent specification (if explicit)
    2. Capability matching (if no target specified)
    3. Agent health status
    """

    def __init__(self):
        self._agents: dict[str, IAgent] = {}
        self._role_index: dict[AgentRole, list[str]] = {}
        self._capability_index: dict[str, list[str]] = {}

    def register(self, agent: IAgent) -> None:
        """Register an agent in the registry."""
        card = agent.agent_card
        self._agents[card.agent_id] = agent

        # Index by role
        if card.role not in self._role_index:
            self._role_index[card.role] = []
        self._role_index[card.role].append(card.agent_id)

        # Index by capabilities
        for cap in card.capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = []
            self._capability_index[cap].append(card.agent_id)

        logger.info(
            f"A2A Registry: Registered '{card.name}' (role={card.role.value}, "
            f"capabilities={card.capabilities})"
        )

    def deregister(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        if agent_id in self._agents:
            card = self._agents[agent_id].agent_card

            # Clean up indices
            if card.role in self._role_index:
                self._role_index[card.role] = [
                    aid for aid in self._role_index[card.role] if aid != agent_id
                ]
            for cap in card.capabilities:
                if cap in self._capability_index:
                    self._capability_index[cap] = [
                        aid for aid in self._capability_index[cap] if aid != agent_id
                    ]

            del self._agents[agent_id]
            logger.info(f"A2A Registry: Deregistered '{card.name}'")

    def discover_by_role(self, role: AgentRole) -> list[AgentCard]:
        """Find agents by role."""
        agent_ids = self._role_index.get(role, [])
        return [self._agents[aid].agent_card for aid in agent_ids if aid in self._agents]

    def discover_by_capability(self, capability: str) -> list[AgentCard]:
        """Find agents by capability."""
        agent_ids = self._capability_index.get(capability, [])
        return [self._agents[aid].agent_card for aid in agent_ids if aid in self._agents]

    def list_all(self) -> list[AgentCard]:
        """List all registered agents."""
        return [agent.agent_card for agent in self._agents.values()]

    def get_agent(self, agent_id: str) -> IAgent | None:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    async def route_task(
        self, task: TaskRequest, context: AgentContext
    ) -> TaskResponse:
        """
        Route a task to the appropriate agent.

        Routing logic:
        1. If target_agent is specified as a role name, route to that role
        2. If not found, try capability-based routing
        3. If still not found, raise TaskRoutingError
        """
        target = task.target_agent

        # Try matching by role name
        if target:
            try:
                role = AgentRole(target)
                agents = self._role_index.get(role, [])
                if agents:
                    agent = self._agents[agents[0]]  # Pick first available
                    logger.info(
                        f"A2A: Routing task {task.task_id} to {agent.agent_card.name} (by role)"
                    )
                    return await agent.process(task, context)
            except ValueError:
                pass

            # Try matching by agent ID
            if target in self._agents:
                agent = self._agents[target]
                return await agent.process(task, context)

        # Capability-based routing: use LLM to match capabilities
        # For now, try each agent until one handles it
        for agent_id, agent in self._agents.items():
            if agent.agent_card.role == AgentRole.ORCHESTRATOR:
                continue  # Don't delegate to the orchestrator

            try:
                result = await agent.process(task, context)
                if result.status == TaskStatus.COMPLETED:
                    return result
            except Exception as e:
                logger.warning(f"Agent {agent_id} failed: {e}")
                continue

        raise TaskRoutingError(
            f"No agent could handle task: {task.query[:100]}"
        )


class A2ABus:
    """
    A2A Communication Bus — Async message passing between agents.

    Supports:
    - Request/response pattern
    - Fire-and-forget pattern
    - Status streaming
    - Event publication
    """

    def __init__(self, registry: AgentRegistry):
        self._registry = registry
        self._task_log: dict[str, TaskResponse] = {}
        self._event_subscribers: dict[str, list] = {}

    async def send_task(
        self, task: TaskRequest, context: AgentContext
    ) -> TaskResponse:
        """Send a task and wait for a response."""
        logger.info(
            f"A2A Bus: Task {task.task_id} from '{task.source_agent}' → '{task.target_agent}'"
        )

        response = await self._registry.route_task(task, context)
        self._task_log[task.task_id] = response

        # Publish task completion event
        await self._publish_event("task_completed", {
            "task_id": task.task_id,
            "status": response.status.value,
            "source": task.source_agent,
            "latency_ms": response.latency_ms,
        })

        return response

    async def send_task_async(
        self, task: TaskRequest, context: AgentContext
    ) -> asyncio.Task:
        """Send a task without waiting (fire-and-forget). Returns an asyncio.Task."""
        return asyncio.create_task(self.send_task(task, context))

    def get_task_status(self, task_id: str) -> TaskResponse | None:
        """Check the status of a previously submitted task."""
        return self._task_log.get(task_id)

    async def _publish_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to subscribers."""
        subscribers = self._event_subscribers.get(event_type, [])
        for callback in subscribers:
            try:
                await callback(data)
            except Exception as e:
                logger.warning(f"Event subscriber error: {e}")

    def subscribe(self, event_type: str, callback) -> None:
        """Subscribe to events."""
        if event_type not in self._event_subscribers:
            self._event_subscribers[event_type] = []
        self._event_subscribers[event_type].append(callback)
