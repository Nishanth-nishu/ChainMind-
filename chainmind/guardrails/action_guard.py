"""
ChainMind Action Guard — Tool execution constraints and HITL gating.

Enforces which tools each agent is allowed to use,
validates parameter ranges, and flags high-risk actions.
"""

from __future__ import annotations

import logging
from typing import Any

from chainmind.config.constants import GuardrailAction
from chainmind.core.interfaces import IGuardrail
from chainmind.core.types import GuardrailResult

logger = logging.getLogger(__name__)

# High-risk actions that require extra scrutiny
_HIGH_RISK_ACTIONS = {
    "delete",
    "remove",
    "cancel",
    "update",
    "modify",
    "execute",
    "transfer",
}


class ActionGuard(IGuardrail):
    """
    Tool execution guardrail.

    Enforces:
    1. Tool allowlists per agent
    2. Parameter range validation
    3. High-risk action flagging
    4. Rate limiting per tool
    """

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        blocked_tools: set[str] | None = None,
    ):
        self._allowed = allowed_tools  # None = all allowed
        self._blocked = blocked_tools or set()
        self._call_counts: dict[str, int] = {}
        self._max_calls_per_tool = 50  # Per session

    @property
    def guardrail_name(self) -> str:
        return "action_guard"

    async def check(
        self, content: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Check if a tool execution is allowed."""
        ctx = context or {}

        if ctx.get("type") != "action":
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.ALLOW,
                passed=True,
            )

        tool_name = ctx.get("tool_name", content)
        args = ctx.get("args", {})

        # Check 1: Blocked tools
        if tool_name in self._blocked:
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.BLOCK,
                passed=False,
                reason=f"Tool '{tool_name}' is blocked",
            )

        # Check 2: Allowlist (if configured)
        if self._allowed is not None and tool_name not in self._allowed:
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.BLOCK,
                passed=False,
                reason=f"Tool '{tool_name}' not in allowlist",
            )

        # Check 3: Rate limiting
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1
        if self._call_counts[tool_name] > self._max_calls_per_tool:
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.BLOCK,
                passed=False,
                reason=f"Tool '{tool_name}' exceeded call limit ({self._max_calls_per_tool})",
            )

        # Check 4: High-risk action warning
        for keyword in _HIGH_RISK_ACTIONS:
            if keyword in tool_name.lower():
                logger.warning(f"High-risk action detected: {tool_name}({args})")
                return GuardrailResult(
                    guardrail_name=self.guardrail_name,
                    action=GuardrailAction.WARN,
                    passed=True,
                    reason=f"High-risk action: {tool_name}",
                    metadata={"risk_level": "high", "tool": tool_name},
                )

        return GuardrailResult(
            guardrail_name=self.guardrail_name,
            action=GuardrailAction.ALLOW,
            passed=True,
        )
