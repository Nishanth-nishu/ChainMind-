"""
EXP003 — Chain-of-Verification (CoVe) Agent
Paper: Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in LLMs"
       Meta AI, 2023. https://arxiv.org/abs/2309.11495

Hypothesis: After the ReAct loop produces a draft answer, generate 2-3
factual verification questions about claims in that answer, answer them
independently, then revise the final answer. Catches molecular hallucinations.
"""
from __future__ import annotations
import time
from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import (
    LLMRequest, LLMMessage, TaskRequest, AgentContext, TaskResponse,
)
from chainmind.config.constants import TaskStatus


PLAN_PROMPT = """\
Given this draft answer to a chemistry question, write 2-3 short factual
verification questions to check its claims (one question per line).

Question: {query}
Draft Answer: {draft}

Verification questions (one per line, each answerable by a Yes/No or a number):"""

VERIFY_PROMPT = """\
Answer this verification question based ONLY on the tool results below.
Do not use your internal knowledge.

Tool Results:
{observations}

Verification Question: {question}
Answer (Yes/No or a number):"""

REFINE_PROMPT = """\
Original Question: {query}
Draft Answer: {draft}

Verification Results:
{verifications}

Based on these verifications, write a corrected final answer.
If all verifications passed, repeat the draft. If any failed, correct them."""


class CoVeAgent(BaseAgent):
    """EXP003: Chain-of-Verification post-processing."""

    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        start = time.perf_counter()
        base_resp = await super().process(task, context)

        if base_resp.status != TaskStatus.COMPLETED or not base_resp.result:
            return base_resp

        observations = "\n".join(
            step.tool_output or ""
            for step in (base_resp.reasoning_trace or [])
            if step.tool_output
        )[:2000]

        # Step 1: Plan verification questions
        try:
            plan_resp = await self._llm_router.generate(LLMRequest(
                messages=[LLMMessage(role="user", content=PLAN_PROMPT.format(
                    query=task.query, draft=base_resp.result[:500]
                ))],
                temperature=0.2, max_tokens=256,
            ))
            questions = [q.strip() for q in plan_resp.content.strip().split("\n")
                         if q.strip() and "?" in q][:3]
        except Exception:
            return base_resp  # CoVe failed → use base

        if not questions:
            return base_resp

        # Step 2: Execute verifications independently
        verifications = []
        for q in questions:
            try:
                v_resp = await self._llm_router.generate(LLMRequest(
                    messages=[LLMMessage(role="user", content=VERIFY_PROMPT.format(
                        observations=observations, question=q
                    ))],
                    temperature=0.1, max_tokens=128,
                ))
                verifications.append(f"Q: {q}\nA: {v_resp.content.strip()}")
            except Exception:
                continue

        # Step 3: Refine final answer
        try:
            refined_resp = await self._llm_router.generate(LLMRequest(
                messages=[LLMMessage(role="user", content=REFINE_PROMPT.format(
                    query=task.query,
                    draft=base_resp.result[:500],
                    verifications="\n\n".join(verifications),
                ))],
                temperature=0.1, max_tokens=512,
            ))
            refined = refined_resp.content.strip()
        except Exception:
            refined = base_resp.result

        elapsed = (time.perf_counter() - start) * 1000
        return TaskResponse(
            task_id=base_resp.task_id,
            source_agent=base_resp.source_agent,
            status=TaskStatus.COMPLETED,
            result=refined,
            reasoning_trace=base_resp.reasoning_trace,
            latency_ms=elapsed,
        )
