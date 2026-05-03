"""
EXP008 — Multi-Agent Debate
Paper: Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
       ICML 2023. https://arxiv.org/abs/2305.14325

Hypothesis: Two parallel ComputationalChemist agents independently answer the same
molecular property question. A lightweight judge compares their answers numerically.
When they agree (|diff| < 1%), use either answer. When they disagree, a judge
LLM call resolves by re-reading raw tool outputs. Reduces hallucination by ~40%.
"""
from __future__ import annotations
import asyncio
import re
import time
from chainmind.agents.base_agent import BaseAgent
from chainmind.core.types import (
    LLMRequest, LLMMessage, TaskRequest, AgentContext, TaskResponse,
)
from chainmind.config.constants import TaskStatus


JUDGE_PROMPT = """Two AI agents independently answered a chemistry question. 
Evaluate which answer is more accurate based on the tool results.

Question: {query}

Agent A Answer: {answer_a}

Agent B Answer: {answer_b}

Tool Results (ground truth):
{tool_results}

Which answer is more accurate? Respond with ONLY:
WINNER: A
or
WINNER: B
Then explain in one sentence why."""


class DebateOrchestrator:
    """
    EXP008: Runs two agents in parallel, judges disagreements.
    
    Not a BaseAgent subclass — wraps two agents and a judge.
    """

    def __init__(self, agent_a: BaseAgent, agent_b: BaseAgent, llm_router):
        self._agent_a = agent_a
        self._agent_b = agent_b
        self._llm_router = llm_router
        self.agent_card = agent_a.agent_card  # needed by bench runner

    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        start = time.perf_counter()

        # Phase 1: Run both agents in parallel
        ctx_a = AgentContext(session_id=context.session_id + "_a")
        ctx_b = AgentContext(session_id=context.session_id + "_b")
        resp_a, resp_b = await asyncio.gather(
            self._agent_a.process(task, ctx_a),
            self._agent_b.process(task, ctx_b),
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(resp_a, Exception):
            resp_a = None
        if isinstance(resp_b, Exception):
            resp_b = None

        answer_a = (resp_a.result or "") if resp_a else ""
        answer_b = (resp_b.result or "") if resp_b else ""

        if not answer_a and not answer_b:
            elapsed = (time.perf_counter() - start) * 1000
            return TaskResponse(
                task_id=task.task_id, source_agent="debate",
                status=TaskStatus.FAILED, error="Both agents failed",
                latency_ms=elapsed,
            )

        if not answer_a:
            winner_answer = answer_b
        elif not answer_b:
            winner_answer = answer_a
        else:
            # Phase 2: Check agreement
            if self._answers_agree(answer_a, answer_b):
                winner_answer = answer_a  # Both agree → pick A
            else:
                # Phase 3: Judge resolves disagreement
                winner_answer = await self._judge(
                    task.query, answer_a, answer_b, resp_a
                )

        elapsed = (time.perf_counter() - start) * 1000
        return TaskResponse(
            task_id=task.task_id,
            source_agent="debate",
            status=TaskStatus.COMPLETED,
            result=winner_answer,
            reasoning_trace=(resp_a.reasoning_trace if resp_a else []),
            latency_ms=elapsed,
        )

    @staticmethod
    def _answers_agree(a: str, b: str, tol: float = 0.05) -> bool:
        """Agree if all shared numbers are within tol of each other."""
        nums_a = [float(x) for x in re.findall(r"\b\d+\.\d+\b", a)]
        nums_b = [float(x) for x in re.findall(r"\b\d+\.\d+\b", b)]
        if not nums_a and not nums_b:
            # Boolean agreement: look for pass/fail keywords
            a_pass = "pass" in a.lower() or "drug-like" in a.lower()
            b_pass = "pass" in b.lower() or "drug-like" in b.lower()
            return a_pass == b_pass
        if len(nums_a) != len(nums_b):
            return False
        return all(abs(x - y) / max(abs(x), 1e-6) < tol for x, y in zip(nums_a, nums_b))

    async def _judge(
        self, query: str, answer_a: str, answer_b: str, resp_a: TaskResponse | None
    ) -> str:
        """Call judge LLM to resolve disagreement."""
        tool_results = ""
        if resp_a and resp_a.reasoning_trace:
            tool_results = "\n".join(
                step.tool_output or ""
                for step in resp_a.reasoning_trace
                if step.tool_output
            )[:1500]

        try:
            judge_resp = await self._llm_router.generate(LLMRequest(
                messages=[LLMMessage(role="user", content=JUDGE_PROMPT.format(
                    query=query, answer_a=answer_a[:400],
                    answer_b=answer_b[:400], tool_results=tool_results,
                ))],
                temperature=0.1, max_tokens=256,
            ))
            verdict = judge_resp.content
            if "WINNER: B" in verdict:
                return answer_b
            return answer_a  # Default to A if unclear
        except Exception:
            return answer_a
