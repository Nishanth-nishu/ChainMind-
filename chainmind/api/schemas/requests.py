"""
ChainMind API Schemas — Request/Response models for the REST API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# === Request Models ===

class QueryRequest(BaseModel):
    """Submit a query to the agentic system."""
    query: str = Field(..., description="The question or task for the AI agents", min_length=1, max_length=10000)
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    target_agent: str | None = Field(None, description="Route to a specific agent role")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    max_steps: int = Field(15, ge=1, le=50, description="Maximum reasoning steps")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")


class KnowledgeIngestRequest(BaseModel):
    """Ingest a document into the knowledge base."""
    content: str = Field(..., description="Document content", min_length=10)
    title: str = Field("", description="Document title")
    category: str = Field("general", description="Document category")
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchRequest(BaseModel):
    """Search the knowledge base."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(5, ge=1, le=50, description="Number of results")
    mode: str = Field("hybrid", description="Search mode: hybrid, vector, bm25")


# === Response Models ===

class QueryResponse(BaseModel):
    """Response from the agentic system."""
    task_id: str
    status: str
    result: str | None = None
    error: str | None = None
    source_agent: str
    latency_ms: float
    reasoning_steps: int = 0
    reasoning_trace: list[dict[str, Any]] = Field(default_factory=list)


class KnowledgeSearchResponse(BaseModel):
    """Knowledge base search results."""
    query: str
    mode: str
    results: list[dict[str, Any]]
    total_results: int


class KnowledgeIngestResponse(BaseModel):
    """Document ingestion result."""
    doc_id: str
    status: str
    title: str


class HealthResponse(BaseModel):
    """System health status."""
    status: str
    is_healthy: bool
    is_ready: bool
    components: dict[str, Any]
    providers: dict[str, Any]


class AgentListResponse(BaseModel):
    """List of registered agents."""
    agents: list[dict[str, Any]]
    total: int


class MetricsResponse(BaseModel):
    """System metrics."""
    counters: dict[str, float]
    gauges: dict[str, float]
    histograms: dict[str, Any]
