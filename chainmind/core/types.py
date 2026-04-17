"""
ChainMind Types — Shared data models used across all components.

Pydantic models for strict validation and type safety.
These are the "contracts" that flow between agents, tools, and providers.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from chainmind.config.constants import (
    AgentRole,
    GuardrailAction,
    ReActStep,
    TaskStatus,
    ToolCategory,
)


# =============================================================================
# LLM Types
# =============================================================================

class LLMMessage(BaseModel):
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    """Request to an LLM provider."""
    messages: list[LLMMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    stop_sequences: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    response_format: dict[str, Any] | None = None  # For structured output


class TokenUsage(BaseModel):
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    usage: TokenUsage
    latency_ms: float
    raw_response: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Agent / A2A Types
# =============================================================================

class AgentCard(BaseModel):
    """Agent identity and capability advertisement (A2A protocol)."""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: AgentRole
    description: str
    capabilities: list[str]
    tools: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    endpoint: str | None = None  # gRPC/HTTP endpoint if remote


class TaskRequest(BaseModel):
    """A2A task request envelope."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: str | None = None
    source_agent: str
    target_agent: str | None = None  # None = orchestrator decides
    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    priority: int = 5  # 1 (highest) to 10 (lowest)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: int = 60


class ReasoningStep(BaseModel):
    """A single step in the ReAct reasoning trace."""
    step_type: ReActStep
    content: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskResponse(BaseModel):
    """A2A task response envelope."""
    task_id: str
    source_agent: str
    status: TaskStatus
    result: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0


# =============================================================================
# MCP Types
# =============================================================================

class MCPToolDefinition(BaseModel):
    """MCP tool definition with JSON Schema for parameters."""
    name: str
    description: str
    category: ToolCategory
    parameters: dict[str, Any]  # JSON Schema for tool parameters
    required_params: list[str] = Field(default_factory=list)


class MCPToolResult(BaseModel):
    """Result from an MCP tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


class AgentContext(BaseModel):
    """Runtime context passed to agents during processing."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_history: list[LLMMessage] = Field(default_factory=list)
    available_tools: list[MCPToolDefinition] = Field(default_factory=list)
    memory_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# (MCPToolDefinition and MCPToolResult moved above AgentContext)


# =============================================================================
# Retrieval Types
# =============================================================================

class RetrievalResult(BaseModel):
    """A single retrieval result from any retriever."""
    doc_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    retriever: str = ""  # Which retriever found this


class RetrievalQuery(BaseModel):
    """Parameters for a retrieval query."""
    query: str
    top_k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)
    mode: str = "hybrid"


# =============================================================================
# Memory Types
# =============================================================================

class MemoryEntry(BaseModel):
    """A single memory entry."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    agent_id: str
    content: str
    memory_type: str = "episodic"  # episodic, semantic, procedural
    importance: float = 0.5  # 0.0 to 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Guardrail Types
# =============================================================================

class GuardrailResult(BaseModel):
    """Result from a guardrail check."""
    guardrail_name: str
    action: GuardrailAction
    passed: bool
    reason: str = ""
    modified_content: str | None = None  # If action is MODIFY
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Health / Observability Types
# =============================================================================

class HealthStatus(BaseModel):
    """Health status for a component."""
    component: str
    healthy: bool
    latency_ms: float = 0.0
    error: str | None = None
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    """A single span in a distributed trace."""
    trace_id: str
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    operation: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"
