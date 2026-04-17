"""
ChainMind Exceptions — Structured exception hierarchy.

All exceptions inherit from ChainMindError for uniform handling.
Each layer has its own exception class for precise error routing.
"""

from __future__ import annotations


class ChainMindError(Exception):
    """Base exception for all ChainMind errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# --- LLM Layer ---

class LLMError(ChainMindError):
    """Base for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Error from a specific LLM provider (API error, auth, etc.)."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(f"[{provider}] {message}", {"provider": provider, "status_code": status_code})
        self.provider = provider
        self.status_code = status_code


class LLMQuotaExhaustedError(LLMProviderError):
    """All API keys for a provider have been rate-limited."""
    pass


class LLMCircuitOpenError(LLMError):
    """Circuit breaker is open — provider is temporarily unavailable."""

    def __init__(self, provider: str):
        super().__init__(
            f"Circuit breaker OPEN for {provider}. Provider temporarily unavailable.",
            {"provider": provider},
        )
        self.provider = provider


class LLMAllProvidersFailedError(LLMError):
    """All providers in the fallback chain have failed."""

    def __init__(self, errors: list[str]):
        super().__init__(
            f"All LLM providers failed: {'; '.join(errors)}",
            {"errors": errors},
        )


# --- Agent Layer ---

class AgentError(ChainMindError):
    """Base for agent-related errors."""
    pass


class AgentExecutionError(AgentError):
    """Agent failed during task execution."""

    def __init__(self, agent_id: str, task_id: str, message: str):
        super().__init__(
            f"Agent {agent_id} failed on task {task_id}: {message}",
            {"agent_id": agent_id, "task_id": task_id},
        )


class AgentTimeoutError(AgentError):
    """Agent exceeded its execution budget."""
    pass


class AgentMaxStepsError(AgentError):
    """Agent hit maximum reasoning steps without resolution."""
    pass


# --- MCP Layer ---

class MCPError(ChainMindError):
    """Base for MCP-related errors."""
    pass


class MCPToolNotFoundError(MCPError):
    """Requested tool does not exist on the MCP server."""
    pass


class MCPToolExecutionError(MCPError):
    """Tool execution failed."""
    pass


# --- Retrieval Layer ---

class RetrievalError(ChainMindError):
    """Base for retrieval-related errors."""
    pass


class IndexingError(RetrievalError):
    """Error during document indexing."""
    pass


# --- Guardrail Layer ---

class GuardrailError(ChainMindError):
    """Base for guardrail errors."""
    pass


class GuardrailBlockedError(GuardrailError):
    """Content was blocked by a guardrail."""

    def __init__(self, guardrail_name: str, reason: str):
        super().__init__(
            f"Blocked by {guardrail_name}: {reason}",
            {"guardrail": guardrail_name, "reason": reason},
        )


# --- A2A Layer ---

class A2AError(ChainMindError):
    """Base for A2A protocol errors."""
    pass


class AgentNotFoundError(A2AError):
    """Requested agent not found in registry."""
    pass


class TaskRoutingError(A2AError):
    """Failed to route task to an appropriate agent."""
    pass
