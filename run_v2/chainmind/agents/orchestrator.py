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


ORCHESTRATOR_SYSTEM_PROMPT = """You are the ChainMind Orchestrator — a strategic planner for Drug Discovery (D4) and general scientific Q&A.

## Your Role
1. Analyze the user's query carefully
2. Think step by step about which specialists are needed
3. Decompose complex queries into focused sub-tasks
4. Route each sub-task to the best specialist agent
5. For general conversation (greetings, personal info, simple questions), answer directly without delegation

## Available Specialist Agents

- computational_chemistry: Molecular analysis (Lipinski, SMILES, PubChem, PDB/MOL2 file parsing, 3D conformers, similarity)
- web_research: Scientific literature search (DuckDuckGo, ArXiv papers, TDC benchmarks)
- knowledge_graph: Visual knowledge graph generation (Mermaid.js) for explaining concepts

## Decomposition Examples

Example 1 — Molecule analysis:
User: "Does Aspirin pass Lipinski's Rule of 5?"
Response: {"analysis": "Direct chemistry question", "sub_tasks": [{"target_agent": "computational_chemistry", "query": "Assess Lipinski Rule of 5 for Aspirin (SMILES: CC(=O)OC1=CC=CC=C1C(=O)O)", "priority": 1}], "requires_aggregation": false}

Example 2 — File parsing:
User: "Parse the mol2 file at /path/to/mol_0001.mol2"
Response: {"analysis": "File analysis request", "sub_tasks": [{"target_agent": "computational_chemistry", "query": "Parse the mol2 file located at /path/to/mol_0001.mol2 and report its properties", "priority": 1}], "requires_aggregation": false}

Example 3 — General conversation (NO delegation):
User: "My name is Nishanth"
Response: {"analysis": "Personal greeting, no specialist needed", "direct_answer": "Nice to meet you, Nishanth! I'm ChainMind, your AI assistant for drug discovery and molecular analysis. How can I help you today?", "sub_tasks": []}

Example 4 — Simple knowledge question (NO delegation):
User: "What is Lipinski's Rule of 5?"
Response: {"analysis": "Simple definition, no computation needed", "direct_answer": "Lipinski's Rule of 5 predicts oral bioavailability. A drug-like molecule should have: MW ≤ 500, LogP ≤ 5, H-bond donors ≤ 5, H-bond acceptors ≤ 10. Violating more than one rule suggests poor absorption.", "sub_tasks": []}

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

If the query is simple, conversational, or doesn't need specialist agents, respond with:
{
    "analysis": "Brief analysis",
    "direct_answer": "Your answer here",
    "sub_tasks": []
}

## Important Rules
- Think step by step before deciding on task decomposition
- NEVER route greetings, personal info, or general chat to specialist agents
- Only route to computational_chemistry when actual molecule analysis or file parsing is needed
- Only route to web_research when the user explicitly asks for papers or literature
- Prefer fewer, focused sub-tasks over many overlapping ones
- Always include the specific data or file paths in each sub-task query
- When a user provides a directory path, pick ONE specific file from it (e.g. append /mol_0000.mol2)
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
        memory_store: Any = None,    # injected for session persistence
    ):
        self._llm_router = llm_router
        self._agent_registry = agent_registry
        self._guardrails = guardrails or []
        self._memory_store = memory_store
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
            # Check for episodic memory
            memory_context = ""
            if self._memory_store:
                if hasattr(self._memory_store, 'retrieve_by_session'):
                    entries = await self._memory_store.retrieve_by_session(context.session_id, top_k=3)
                else:
                    entries = await self._memory_store.retrieve(context.session_id, top_k=3)
                if entries:
                    memory_context = "\n".join("- " + e.content for e in entries)

            # Step 1: Decompose the query
            decomposition = await self._decompose_query(task.query, memory_context)
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

            # Store orchestrator's decision in memory
            if self._memory_store:
                from chainmind.core.types import MemoryEntry
                entry = MemoryEntry(
                    session_id=context.session_id,
                    agent_id=self._agent_id,
                    content=f"Q: {task.query}\nA: {aggregated}",
                    memory_type="episodic",
                )
                await self._memory_store.store(entry)

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

    async def _decompose_query(self, query: str, memory_context: str = "") -> dict[str, Any]:
        """Use LLM to decompose a query into sub-tasks."""
        
        content = query
        if memory_context:
            content = f"Previous Conversation Context:\n{memory_context}\n\nCurrent Query: {query}"

        response = await self._llm_router.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content=content)],
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
                system_prompt="You are ChainMind, a helpful AI assistant for drug discovery and molecular science. Answer the question conversationally and helpfully.",
                temperature=0.5,
                max_tokens=1024,
            )
        )
        return response.content
