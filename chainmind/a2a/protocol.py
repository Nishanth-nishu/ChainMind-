"""
chainmind/a2a/protocol.py
Agent-to-Agent (A2A) protocol: AgentRegistry and A2ABus.

The A2A protocol defines how agents discover and delegate tasks to each other.
Each agent advertises an AgentCard (capability advertisement).
The AgentRegistry matches task target_agent strings to registered agents.
"""
from __future__ import annotations

import logging
from typing import Any

from chainmind.core.types import AgentContext, AgentCard, TaskRequest, TaskResponse
from chainmind.config.constants import TaskStatus

logger = logging.getLogger(__name__)


# Role → agent name aliases accepted in task routing
_ROLE_ALIASES: dict[str, list[str]] = {
    "computational_chemistry": [
        "computational_chemistry", "computational_chemist",
        "computationalchemistagent", "chemist", "molecular",
        "reflexionagent", "selfconsistencyagent", "coveagent",
        "fewshotchemistagent", "toolragagent", "structuredoutputagent",
        "chemragagent",
    ],
    "web_research": [
        "web_research", "web_researcher", "webresearchagent",
        "research", "literature",
    ],
    "knowledge_graph": [
        "knowledge_graph", "knowledge_graph_agent", "knowledgegraphagent",
        "graph", "kg",
    ],
}


class AgentRegistry:
    """
    Registry of specialist agents, keyed by their AgentCard role.

    Usage:
        reg = AgentRegistry()
        reg.register(ComputationalChemistAgent(...))
        reg.register(WebResearchAgent(...))
        response = await reg.route_task(task_request, context)
    """

    def __init__(self):
        self._agents: dict[str, Any] = {}  # role_key → agent

    def register(self, agent: Any) -> None:
        """Register an agent using its AgentCard role as the routing key."""
        try:
            card: AgentCard = agent.agent_card
            role_key = card.role.value.lower()
            self._agents[role_key] = agent
            logger.info(f"A2A: Registered '{card.name}' as '{role_key}'")
        except Exception as e:
            logger.warning(f"A2A: Failed to register agent: {e}")

    def get_agent(self, target: str) -> Any | None:
        """Look up an agent by role alias."""
        t_lower = target.lower().replace("-", "_").replace(" ", "_")

        # Direct match on role key
        if t_lower in self._agents:
            return self._agents[t_lower]

        # Alias match
        for role_key, aliases in _ROLE_ALIASES.items():
            if t_lower in aliases:
                return self._agents.get(role_key)

        # Partial match fallback (e.g. "chemistry" → "computational_chemistry")
        for role_key, agent in self._agents.items():
            if t_lower in role_key or role_key in t_lower:
                return agent

        # Last resort: check agent names
        for role_key, agent in self._agents.items():
            try:
                if t_lower in agent.agent_card.name.lower():
                    return agent
            except Exception:
                pass

        return None

    async def route_task(
        self, task: TaskRequest, context: AgentContext
    ) -> TaskResponse:
        """Route a task to the appropriate registered agent."""
        target = task.target_agent or ""
        agent = self.get_agent(target)

        if agent is None:
            # Fallback: use first registered agent (usually the specialist)
            if self._agents:
                agent = next(iter(self._agents.values()))
                logger.warning(
                    f"A2A: No agent found for '{target}', "
                    f"falling back to '{list(self._agents.keys())[0]}'"
                )
            else:
                return TaskResponse(
                    task_id=task.task_id,
                    source_agent="a2a_registry",
                    status=TaskStatus.FAILED,
                    error=f"No agent registered for '{target}' and registry is empty.",
                )

        logger.debug(f"A2A: Routing task '{task.task_id}' → {agent.agent_card.name}")
        return await agent.process(task, context)

    @property
    def registered_agents(self) -> list[str]:
        return list(self._agents.keys())


class A2ABus:
    """
    Message bus for broadcasting tasks to multiple agents.
    Used in api/app.py for coordinating session-level operations.
    """

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    async def broadcast(
        self, task: TaskRequest, context: AgentContext
    ) -> list[TaskResponse]:
        """Send a task to all registered agents and collect responses."""
        responses = []
        for role_key, agent in self._registry._agents.items():
            try:
                resp = await agent.process(task, context)
                responses.append(resp)
            except Exception as e:
                logger.error(f"A2ABus: Agent '{role_key}' failed: {e}")
        return responses

    async def route(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        """Route to the best agent (delegates to AgentRegistry)."""
        return await self._registry.route_task(task, context)
