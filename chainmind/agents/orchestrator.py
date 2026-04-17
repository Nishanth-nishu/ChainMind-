"""
ChainMind Orchestrator Agent — Strategic planner and task delegator.

Implements the Orchestrator-Workers pattern:
1. Receives user query
2. Decomposes into sub-tasks
3. Routes sub-tasks to specialist agents via A2A protocol
4. Aggregates results with conflict resolution
5. Provides unified response

This is NOT a ReAct agent — it's a strategic planner that delegates
to ReAct-based worker agents.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from chainmind.config.constants import AgentRole, TaskStatus
from chainmind.core.interfaces import IAgent, IGuardrail
from chainmind.core.types import (
    AgentCard,
    AgentContext,
    LLMMessage,
    LLMRequest,
    ReasoningStep,
    TaskRequest,
    TaskResponse,
)
from chainmind.config.constants import ReActStep
from chainmind.llm.router import LLMRouter

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """You are the ChainMind Orchestrator — a strategic planner for supply chain operations.

Your role is to:
1. Analyze the user's query
2. Determine which specialist agents should handle it
3. Decompose complex queries into sub-tasks for each specialist
4. Aggregate their results into a unified, actionable response

## Available Specialist Agents
- demand_forecasting: Demand predictions, trend analysis, seasonal patterns
- inventory_management: Stock levels, reorder points, warehouse optimization
- procurement: Supplier management, purchase orders, cost optimization
- logistics: Shipping, routing, delivery tracking, transportation
- quality_assurance: Quality metrics, anomaly detection, compliance

## Response Format
Always respond with valid JSON:
{
    "analysis": "Brief analysis of the query",
    "sub_tasks": [
        {
            "target_agent": "agent_role",
            "query": "specific sub-task query",
            "priority": 1-10
        }
    ],
    "requires_aggregation": true/false
}

If the query is simple and doesn't need specialist agents, respond with:
{
    "analysis": "Brief analysis",
    "direct_answer": "Your answer here",
    "sub_tasks": []
}
"""


class OrchestratorAgent(IAgent):
    """
    Strategic orchestrator that decomposes and delegates tasks.

    Does not use the ReAct loop — instead uses structured LLM calls
    for task decomposition and result aggregation.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        agent_registry: Any = None,  # A2A registry, injected later
        guardrails: list[IGuardrail] | None = None,
    ):
        self._llm_router = llm_router
        self._agent_registry = agent_registry
        self._guardrails = guardrails or []
        self._agent_id = "orchestrator-001"

    @property
    def agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self._agent_id,
            name="ChainMind Orchestrator",
            role=AgentRole.ORCHESTRATOR,
            description="Strategic planner that decomposes queries and delegates to specialist agents",
            capabilities=[
                "task_decomposition",
                "agent_routing",
                "result_aggregation",
                "conflict_resolution",
            ],
        )

    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        """
        Process a query by decomposing and delegating to specialists.
        """
        start_time = time.perf_counter()
        reasoning_trace: list[ReasoningStep] = []

        try:
            # Step 1: Decompose the query
            decomposition = await self._decompose_query(task.query)
            reasoning_trace.append(ReasoningStep(
                step_type=ReActStep.THINK,
                content=f"Decomposition: {json.dumps(decomposition, indent=2)}",
            ))

            # Check for direct answer (no delegation needed)
            if decomposition.get("direct_answer"):
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return TaskResponse(
                    task_id=task.task_id,
                    source_agent=self._agent_id,
                    status=TaskStatus.COMPLETED,
                    result=decomposition["direct_answer"],
                    reasoning_trace=reasoning_trace,
                    latency_ms=elapsed_ms,
                )

            # Step 2: Route sub-tasks to specialists
            sub_tasks = decomposition.get("sub_tasks", [])
            sub_results: list[TaskResponse] = []

            for sub_task_spec in sub_tasks:
                target_role = sub_task_spec.get("target_agent", "")
                sub_query = sub_task_spec.get("query", task.query)

                reasoning_trace.append(ReasoningStep(
                    step_type=ReActStep.ACT,
                    content=f"Delegating to {target_role}: {sub_query}",
                ))

                # Execute via A2A if registry is available
                if self._agent_registry:
                    sub_task = TaskRequest(
                        parent_task_id=task.task_id,
                        source_agent=self._agent_id,
                        target_agent=target_role,
                        query=sub_query,
                        context=task.context,
                    )
                    result = await self._agent_registry.route_task(sub_task, context)
                    sub_results.append(result)

                    reasoning_trace.append(ReasoningStep(
                        step_type=ReActStep.OBSERVE,
                        content=f"Result from {target_role}: {result.status.value} - {result.result or result.error}",
                    ))

            # Step 3: Aggregate results
            if sub_results:
                aggregated = await self._aggregate_results(
                    task.query, sub_results, decomposition.get("analysis", "")
                )
            else:
                # No agents available — answer directly
                aggregated = await self._direct_answer(task.query)

            reasoning_trace.append(ReasoningStep(
                step_type=ReActStep.VERIFY,
                content="Results aggregated successfully",
            ))

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return TaskResponse(
                task_id=task.task_id,
                source_agent=self._agent_id,
                status=TaskStatus.COMPLETED,
                result=aggregated,
                reasoning_trace=reasoning_trace,
                latency_ms=elapsed_ms,
                data={"sub_task_count": len(sub_results)},
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            return TaskResponse(
                task_id=task.task_id,
                source_agent=self._agent_id,
                status=TaskStatus.FAILED,
                error=str(e),
                reasoning_trace=reasoning_trace,
                latency_ms=elapsed_ms,
            )

    async def _decompose_query(self, query: str) -> dict[str, Any]:
        """Use LLM to decompose a query into sub-tasks."""
        response = await self._llm_router.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=query)],
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback: treat entire query as a direct answer request
            return {
                "analysis": "Could not decompose — answering directly",
                "direct_answer": None,
                "sub_tasks": [],
            }

    async def _aggregate_results(
        self, original_query: str, results: list[TaskResponse], analysis: str
    ) -> str:
        """Aggregate sub-task results into a unified response."""
        results_text = "\n\n".join(
            f"**{r.source_agent}** ({r.status.value}):\n{r.result or r.error}"
            for r in results
        )

        aggregate_prompt = f"""Combine these specialist agent results into a unified, actionable response.

Original Query: {original_query}
Analysis: {analysis}

Specialist Results:
{results_text}

Provide a comprehensive, well-structured response that synthesizes all findings.
If there are conflicting recommendations, note them and suggest the best course of action."""

        response = await self._llm_router.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=aggregate_prompt)],
                temperature=0.3,
                max_tokens=2048,
            )
        )
        return response.content

    async def _direct_answer(self, query: str) -> str:
        """Answer directly when no specialists are available."""
        response = await self._llm_router.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=query)],
                system_prompt="You are a supply chain AI assistant. Answer the question based on your knowledge.",
                temperature=0.5,
                max_tokens=2048,
            )
        )
        return response.content
