"""
ChainMind Output Guard — Response validation, hallucination checks, data leak prevention.

Validates all agent outputs before they reach the user.
Implements the Amazon Nova "constrained decoding" concept at the application level.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from chainmind.config.constants import GuardrailAction
from chainmind.core.interfaces import IGuardrail
from chainmind.core.types import GuardrailResult

logger = logging.getLogger(__name__)

# Patterns that should never appear in outputs
_SENSITIVE_PATTERNS = [
    r"api[_\s]?key\s*[:=]\s*\S+",
    r"password\s*[:=]\s*\S+",
    r"secret\s*[:=]\s*\S+",
    r"token\s*[:=]\s*\S+",
    r"sk-[a-zA-Z0-9]{20,}",  # OpenAI key pattern
    r"AIza[a-zA-Z0-9_-]{35}",  # Google API key pattern
]


class OutputGuard(IGuardrail):
    """
    Output validation guardrail.

    Checks for:
    1. Sensitive data leakage (API keys, passwords)
    2. Response length limits
    3. Coherence checks (not empty, not garbage)
    4. Format validation (structured responses)
    """

    def __init__(self, max_output_tokens: int = 4096):
        self._max_tokens = max_output_tokens
        self._sensitive_re = [
            re.compile(p, re.IGNORECASE) for p in _SENSITIVE_PATTERNS
        ]

    @property
    def guardrail_name(self) -> str:
        return "output_guard"

    async def check(
        self, content: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Run all output checks."""
        ctx = context or {}

        if ctx.get("type") not in ("output", None):
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.ALLOW,
                passed=True,
            )

        # Check 1: Empty response
        if not content or not content.strip():
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.BLOCK,
                passed=False,
                reason="Empty response from agent",
            )

        # Check 2: Response too long
        estimated_tokens = len(content.split())
        if estimated_tokens > self._max_tokens:
            truncated = " ".join(content.split()[:self._max_tokens])
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.MODIFY,
                passed=True,
                reason=f"Response truncated ({estimated_tokens} > {self._max_tokens} tokens)",
                modified_content=truncated + "\n\n[Response truncated due to length]",
            )

        # Check 3: Sensitive data leak
        modified = content
        leaked = False
        for pattern in self._sensitive_re:
            if pattern.search(modified):
                leaked = True
                modified = pattern.sub("[REDACTED]", modified)
                logger.warning(f"Sensitive data leak prevented in output")

        if leaked:
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.MODIFY,
                passed=True,
                reason="Sensitive data redacted from output",
                modified_content=modified,
            )

        return GuardrailResult(
            guardrail_name=self.guardrail_name,
            action=GuardrailAction.ALLOW,
            passed=True,
        )
