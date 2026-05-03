"""
EXP002 — Self-Consistency Sampling Agent
Paper: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in LLMs"
       ICML 2022. https://arxiv.org/abs/2203.11171

Hypothesis: Sample N=3 responses at T=0.7, extract numerical/boolean answers,
pick the majority. Reduces variance on deterministic molecular property tasks.
"""
from __future__ import annotations
import re
from collections import Counter
from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import LLMRequest, LLMMessage, TaskRequest, AgentContext, TaskResponse
from chainmind.config.constants import TaskStatus
import time


class SelfConsistencyAgent(BaseAgent):
    """EXP002: N=3 sample majority vote on final synthesis."""

    N_SAMPLES = 3
    SAMPLE_TEMP = 0.7

    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        """Run base ReAct once to get tool results, then sample N answers."""
        start = time.perf_counter()
        # Phase 1: run normal ReAct to collect tool observations
        base_resp = await super().process(task, context)

        if base_resp.status != TaskStatus.COMPLETED:
            return base_resp  # propagate failures

        # Phase 2: self-consistency — sample N final answers and vote
        observations = "\n".join(
            step.tool_output or step.content
            for step in (base_resp.reasoning_trace or [])
            if step.step_type.value == "observe" and step.tool_output
        )

        if not observations:
            return base_resp  # no observations → use base answer

        synthesis_prompt = f"""Based on these tool results, answer the question.
Question: {task.query}
Tool Results:
{observations[:2000]}

Provide a concise, factual answer based ONLY on the tool results above."""

        answers = []
        for _ in range(self.N_SAMPLES):
            try:
                resp = await self._llm_router.generate(LLMRequest(
                    messages=[LLMMessage(role="user", content=synthesis_prompt)],
                    temperature=self.SAMPLE_TEMP,
                    max_tokens=512,
                ))
                answers.append(resp.content)
            except Exception:
                answers.append(base_resp.result or "")

        # Vote: pick answer whose key numbers/booleans appear most often
        voted = self._majority_vote(answers)
        elapsed = (time.perf_counter() - start) * 1000

        return TaskResponse(
            task_id=base_resp.task_id,
            source_agent=base_resp.source_agent,
            status=TaskStatus.COMPLETED,
            result=voted,
            reasoning_trace=base_resp.reasoning_trace,
            latency_ms=elapsed,
        )

    def _majority_vote(self, answers: list[str]) -> str:
        """Vote on numerical/boolean values across N answers."""
        if not answers:
            return ""

        # Extract all floating point numbers from each answer
        def extract_nums(text):
            return tuple(round(float(x), 1) for x in re.findall(r"\b\d+\.\d+\b", text))

        num_groups = [extract_nums(a) for a in answers]
        # If any two agree on same number set → pick that answer
        count = Counter(num_groups)
        most_common_nums, _ = count.most_common(1)[0]

        for answer, nums in zip(answers, num_groups):
            if nums == most_common_nums:
                return answer

        return answers[0]  # fallback: first answer
