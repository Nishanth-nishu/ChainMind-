"""
chainmind/core/exceptions.py
All domain exceptions for ChainMind.
"""


# ---------------------------------------------------------------------------
# Agent exceptions
# ---------------------------------------------------------------------------

class ChainMindError(Exception):
    """Base exception for all ChainMind errors."""


class AgentExecutionError(ChainMindError):
    """Raised when an agent encounters a fatal execution error."""


class AgentMaxStepsError(AgentExecutionError):
    """Raised when an agent exceeds its maximum step budget."""


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent exceeds its time budget."""


class GuardrailBlockedError(ChainMindError):
    """Raised when a guardrail blocks a request."""
    def __init__(self, guardrail_name: str, reason: str):
        self.guardrail_name = guardrail_name
        self.reason = reason
        super().__init__(f"Guardrail '{guardrail_name}' blocked: {reason}")


# ---------------------------------------------------------------------------
# LLM exceptions
# ---------------------------------------------------------------------------

class LLMError(ChainMindError):
    """Base LLM exception."""


class LLMProviderError(LLMError):
    """A specific provider failed to generate a response."""


class LLMQuotaExhaustedError(LLMError):
    """All API keys for a provider are exhausted."""


class LLMCircuitOpenError(LLMError):
    """Circuit breaker is open for this provider."""


class LLMAllProvidersFailedError(LLMError):
    """All providers in the fallback chain failed."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("All LLM providers failed:\n" + "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# MCP exceptions
# ---------------------------------------------------------------------------

class MCPError(ChainMindError):
    """Base MCP exception."""


class MCPToolNotFoundError(MCPError):
    """Tool not found in MCP server registry."""


class MCPToolExecutionError(MCPError):
    """Tool raised an exception during execution."""


# ---------------------------------------------------------------------------
# A2A exceptions
# ---------------------------------------------------------------------------

class A2AError(ChainMindError):
    """Base A2A protocol exception."""


class A2AAgentNotFoundError(A2AError):
    """No agent registered for the requested capability."""


# ---------------------------------------------------------------------------
# Memory exceptions
# ---------------------------------------------------------------------------

class MemoryError(ChainMindError):
    """Base memory exception."""


class MemoryStoreError(MemoryError):
    """Failed to store a memory entry."""


class MemoryRetrieveError(MemoryError):
    """Failed to retrieve memory entries."""
