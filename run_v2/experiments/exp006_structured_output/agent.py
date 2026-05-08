"""
EXP006 — Structured Output Enforcement
Paper: Willard & Louf, "Efficient Guided Generation for Large Language Models"
       (Outlines framework), 2023. https://arxiv.org/abs/2307.09702
       + OpenAI JSON mode best practices

Hypothesis: Current tool-call parsing uses regex over free-form LLM text —
fragile with 7B models. Reformatting the THINK prompt to mandate JSON output
and adding a JSON repair layer eliminates parsing errors entirely.
"""
from __future__ import annotations
import json
import re
from chainmind.agents.specialists import ComputationalChemistAgent as BaseAgent
from chainmind.core.types import LLMMessage


JSON_THINK_SUFFIX = """
RESPONSE FORMAT (MANDATORY — respond with ONLY one of these two JSON objects):

If calling a tool:
{"thought": "...", "action": "tool_name", "action_input": {...}}

If answering directly:
{"thought": "...", "final_answer": "..."}

Respond with valid JSON only. No markdown, no extra text."""


class StructuredOutputAgent(BaseAgent):
    """EXP006: Enforced JSON output + schema validation + repair."""

    def _build_think_prompt(
        self,
        messages: list[LLMMessage],
        tool_descriptions: str,
        memory_context: str,
        step: int,
    ) -> str:
        base = super()._build_think_prompt(
            messages, tool_descriptions, memory_context, step
        )
        # Strip the free-form format instructions and replace with JSON mandate
        base = re.sub(
            r"If you need to use a tool.*?Respond now:",
            JSON_THINK_SUFFIX,
            base,
            flags=re.DOTALL,
        )
        return base

    def _parse_tool_call(self, thought: str):
        """Override: parse from enforced JSON instead of regex."""
        parsed = self._repair_and_parse_json(thought)

        if parsed is None:
            return super()._parse_tool_call(thought)  # fallback: regex

        if "action" in parsed and "action_input" in parsed:
            return parsed["action"], parsed.get("action_input", {})

        if "final_answer" in parsed:
            return None  # signal: this is a final_answer block, not a tool call

        # Parsed something but missing both keys (e.g. extracted ACTION_INPUT sub-object)
        # → fall back to regex-based parsing
        return super()._parse_tool_call(thought)


    def _is_final_answer(self, thought: str) -> bool:
        parsed = self._repair_and_parse_json(thought)
        if parsed and "final_answer" in parsed:
            return True
        return "FINAL_ANSWER:" in thought  # fallback

    def _extract_final_answer(self, thought: str) -> str:
        parsed = self._repair_and_parse_json(thought)
        if parsed and "final_answer" in parsed:
            return parsed["final_answer"]
        return super()._extract_final_answer(thought)

    @staticmethod
    def _repair_and_parse_json(text: str) -> dict | None:
        """Try to parse JSON; attempt common repairs if it fails."""
        # 1. Direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 2. Extract first JSON object via brace matching
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Try replacing single quotes with double
                        try:
                            return json.loads(candidate.replace("'", '"'))
                        except Exception:
                            return None
        return None
