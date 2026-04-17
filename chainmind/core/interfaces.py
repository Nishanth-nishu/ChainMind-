"""
ChainMind Interfaces — Abstract base classes for SOLID compliance.

Dependency Inversion Principle: All components depend on these abstractions,
not on concrete implementations. This enables swappability of LLM providers,
retrievers, agents, and tools without modifying calling code.
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
    RetrievalResult,
    TaskRequest,
    TaskResponse,
)


# =============================================================================
# LLM Provider Interface (Strategy Pattern)
# =============================================================================

class ILLMProvider(ABC):
    """Abstract interface for LLM providers (Gemini, OpenAI, Ollama)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier."""
        ...

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion from the LLM."""
        ...

    @abstractmethod
    async def generate_structured(
        self, request: LLMRequest, schema: dict[str, Any]
    ) -> LLMResponse:
        """Generate a structured (JSON schema-constrained) response."""
        ...

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available."""
        ...


# =============================================================================
# Retriever Interface (Strategy Pattern for RAG)
# =============================================================================

class IRetriever(ABC):
    """Abstract interface for retrieval strategies."""

    @property
    @abstractmethod
    def retriever_name(self) -> str:
        """Return retriever identifier (bm25, dense, splade, hybrid)."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        ...

    @abstractmethod
    async def index(self, documents: list[dict[str, Any]]) -> int:
        """Index documents. Returns count of documents indexed."""
        ...


# =============================================================================
# Agent Interface
# =============================================================================

class IAgent(ABC):
    """Abstract interface for all agents (orchestrator and specialists)."""

    @property
    @abstractmethod
    def agent_card(self) -> AgentCard:
        """Return agent's identity and capabilities."""
        ...

    @abstractmethod
    async def process(self, task: TaskRequest, context: AgentContext) -> TaskResponse:
        """Process a task request and return a response."""
        ...


# =============================================================================
# MCP Server Interface
# =============================================================================

class IMCPServer(ABC):
    """Abstract interface for MCP tool servers."""

    @property
    @abstractmethod
    def server_name(self) -> str:
        """Return server identifier."""
        ...

    @abstractmethod
    def list_tools(self) -> list[MCPToolDefinition]:
        """List all available tools with their schemas."""
        ...

    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute a named tool with the given arguments."""
        ...


# =============================================================================
# Guardrail Interface
# =============================================================================

class IGuardrail(ABC):
    """Abstract interface for safety guardrails."""

    @property
    @abstractmethod
    def guardrail_name(self) -> str:
        """Return guardrail identifier."""
        ...

    @abstractmethod
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """Check content against guardrail rules."""
        ...


# =============================================================================
# Memory Store Interface
# =============================================================================

class IMemoryStore(ABC):
    """Abstract interface for memory storage backends."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns entry ID."""
        ...

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve relevant memories."""
        ...

    @abstractmethod
    async def clear(self, session_id: str | None = None) -> int:
        """Clear memories. Returns count of entries cleared."""
        ...
