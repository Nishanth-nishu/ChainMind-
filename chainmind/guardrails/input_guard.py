"""
ChainMind Input Guard — Prompt injection detection, PII filtering, input validation.

First line of defense — sanitizes all inputs before they reach agents.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from chainmind.config.constants import GuardrailAction
from chainmind.core.interfaces import IGuardrail
from chainmind.core.types import GuardrailResult

logger = logging.getLogger(__name__)

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?above",
    r"you\s+are\s+now\s+in\s+.*mode",
    r"system\s*:\s*override",
    r"forget\s+everything",
    r"new\s+instructions?\s*:",
    r"<\s*system\s*>",
    r"\{\{.*system.*\}\}",
    r"jailbreak",
    r"dan\s+mode",
]

# PII patterns
_PII_PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
}


class InputGuard(IGuardrail):
    """
    Input sanitization guardrail.

    Checks for:
    1. Prompt injection attempts
    2. PII in input (redacts automatically)
    3. Input length limits
    4. Malformed input
    """

    def __init__(self, max_input_length: int = 10000, block_pii: bool = True):
        self._max_length = max_input_length
        self._block_pii = block_pii
        self._injection_re = [
            re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS
        ]

    @property
    def guardrail_name(self) -> str:
        return "input_guard"

    async def check(
        self, content: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Run all input checks."""
        ctx = context or {}

        # Only apply to input-type checks
        if ctx.get("type") not in ("input", None):
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.ALLOW,
                passed=True,
            )

        # Check 1: Length
        if len(content) > self._max_length:
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.BLOCK,
                passed=False,
                reason=f"Input exceeds maximum length ({len(content)} > {self._max_length})",
            )

        # Check 2: Prompt injection
        for pattern in self._injection_re:
            if pattern.search(content):
                logger.warning(f"Prompt injection detected: {pattern.pattern}")
                return GuardrailResult(
                    guardrail_name=self.guardrail_name,
                    action=GuardrailAction.BLOCK,
                    passed=False,
                    reason="Potential prompt injection detected",
                )

        # Check 3: PII redaction
        modified_content = content
        pii_found = False
        if self._block_pii:
            for pii_type, pattern in _PII_PATTERNS.items():
                matches = re.findall(pattern, modified_content)
                if matches:
                    pii_found = True
                    modified_content = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", modified_content)
                    logger.info(f"PII redacted: {pii_type} ({len(matches)} instances)")

        if pii_found:
            return GuardrailResult(
                guardrail_name=self.guardrail_name,
                action=GuardrailAction.MODIFY,
                passed=True,
                reason="PII detected and redacted",
                modified_content=modified_content,
            )

        return GuardrailResult(
            guardrail_name=self.guardrail_name,
            action=GuardrailAction.ALLOW,
            passed=True,
        )
