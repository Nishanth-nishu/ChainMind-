"""
EXP001 — Enhanced Reflexion Agent
Paper: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"
       NeurIPS 2023. https://arxiv.org/abs/2303.11366

Hypothesis
----------
The current _reflect() generates a verbal reflection on tool failure, but the
result is stored only in the reasoning trace — it is NEVER re-injected into
the THINK prompt for the next iteration. This means the agent ignores its own
lessons. By maintaining a verbal reflection buffer and prepending it to every
subsequent THINK step, the agent can learn within a single episode.

Change from Baseline
--------------------
- _reflect(): stores the reflection string in self._reflection_buffer
- _build_think_prompt(): prepends reflection_buffer to the prompt

Expected Gain: +10-15% TSR on Cat-D (multi-step), +8% on Cat-A
"""

from __future__ import annotations

from typing import Any

from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import (
    AgentContext, AgentCard, LLMMessage, LLMRequest,
    MCPToolResult, ReasoningStep, TaskRequest, TaskResponse,
)
from chainmind.config.constants import AgentRole, ReActStep


class ReflexionAgent(BaseAgent):
    """
    Variant 1: Enhanced Reflexion with verbal episodic buffer.

    Key difference from BaseAgent:
    - Maintains a _reflection_buffer (list of past reflections)
    - Prepends the buffer to every THINK prompt so the agent avoids
      repeating mistakes within the same episode
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reflection_buffer: list[str] = []

    async def _reflect(
        self, thought: str, tool_name: str, result: MCPToolResult, query: str
    ) -> str:
        """Override: store reflection in episodic buffer (Reflexion pattern)."""
        reflect_prompt = f"""The previous action produced unsatisfactory results.

Original query: {query}
Tool used: {tool_name}
Result: {result.error or str(result.result)[:300]}

Write a concise lesson (1-2 sentences) for what to do differently next time:"""

        try:
            response = await self._llm_router.generate(
                LLMRequest(
                    messages=[LLMMessage(role="user", content=reflect_prompt)],
                    temperature=0.2,
                    max_tokens=256,
                )
            )
            reflection = response.content.strip()
        except Exception as e:
            reflection = f"Tool {tool_name} failed. Try a different approach."

        # KEY CHANGE: store in buffer so it feeds into next THINK step
        self._reflection_buffer.append(f"[Lesson from step]: {reflection}")
        return reflection

    def _build_think_prompt(
        self,
        messages: list[LLMMessage],
        tool_descriptions: str,
        memory_context: str,
        step: int,
    ) -> str:
        """Override: inject reflection buffer before THINK instructions."""
        base_prompt = super()._build_think_prompt(
            messages, tool_descriptions, memory_context, step
        )

        if not self._reflection_buffer:
            return base_prompt

        # Prepend accumulated reflections (last 3 only to avoid bloat)
        recent = self._reflection_buffer[-3:]
        buffer_text = "\n".join(recent)
        reflexion_block = (
            f"\n## Lessons from Previous Attempts (Reflexion Buffer)\n"
            f"{buffer_text}\n"
            f"Apply these lessons when deciding your next action.\n"
        )

        # Insert after the Available Tools section
        return base_prompt.replace(
            "## Conversation History",
            f"{reflexion_block}\n## Conversation History",
        )
