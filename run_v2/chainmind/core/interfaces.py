"""
chainmind/core/interfaces.py
Abstract base interfaces for all pluggable ChainMind components.

These ABCs define the contracts between:
- Agents ↔ Orchestrator (IAgent)
- Agents ↔ Tools (IMCPServer)
- Agents ↔ Memory (IMemoryStore)
- Agents ↔ Safety (IGuardrail)
- Router ↔ LLM backends (ILLMProvider)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from chainmind.core.types import (
    AgentCard,
    AgentContext,
    GuardrailResult,
    LLMRequest,
    LLMResponse,
    MCPToolDefinition,
    MCPToolResult,
    MemoryEntry,
    TaskRequest,
    TaskResponse,
)


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------

class IAgent(ABC):
    """Contract for all ChainMind agents (specialist and orchestrator)."""

    @property
    @abstractmethod
    def agent_card(self) -> AgentCard:
        """Return this agent's identity + capability advertisement."""
        ...

    @abstractmethod
    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        """Process a task and return a response."""
        ...


# ---------------------------------------------------------------------------
# MCP Server interface
# ---------------------------------------------------------------------------

class IMCPServer(ABC):
    """Contract for Model Context Protocol tool servers."""

    @abstractmethod
    def list_tools(self) -> list[MCPToolDefinition]:
        """Return all tools exposed by this server."""
        ...

    @abstractmethod
    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> MCPToolResult:
        """Execute a named tool with the given arguments."""
        ...


# ---------------------------------------------------------------------------
# Memory Store interface
# ---------------------------------------------------------------------------

class IMemoryStore(ABC):
    """Contract for short-term and long-term memory backends."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Persist a memory entry."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve the most relevant memory entries for a query."""
        ...

    async def retrieve_by_session(
        self, session_id: str, top_k: int = 5
    ) -> list[MemoryEntry]:
        """Retrieve entries by session ID (optional override)."""
        return await self.retrieve(session_id, top_k=top_k)


# ---------------------------------------------------------------------------
# Guardrail interface
# ---------------------------------------------------------------------------

class IGuardrail(ABC):
    """Contract for input/output/action safety guardrails."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique guardrail identifier."""
        ...

    @abstractmethod
    async def check(
        self, content: str, metadata: dict[str, Any]
    ) -> GuardrailResult:
        """
        Check content against this guardrail.

        Args:
            content:  The text or tool name being checked.
            metadata: Context dict, e.g. {"type": "input"/"output"/"action"}.

        Returns:
            GuardrailResult with passed=True/False and optional modified_content.
        """
        ...


# ---------------------------------------------------------------------------
# LLM Provider interface
# ---------------------------------------------------------------------------

class ILLMProvider(ABC):
    """Contract for LLM inference backends (local vLLM, OpenAI, etc.)."""

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for the given request."""
        ...

    async def generate_structured(
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> LLMResponse:
        """Generate a structured (JSON-schema-constrained) completion."""
        # Default: fall back to standard generate (providers override if supported)
        return await self.generate(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream tokens from the provider."""
        response = await self.generate(request)
        yield response.content

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and healthy."""
        ...
