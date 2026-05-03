from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from chainmind.config.constants import TaskStatus, ReActStep

class LLMMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class LLMRequest(BaseModel):
    messages: List[LLMMessage]
    temperature: float = 0.0
    max_tokens: int = 2048

class LLMResponse(BaseModel):
    content: str
    usage: Optional[Dict[str, int]] = None

class MCPToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class MCPToolResult(BaseModel):
    output: str
    error: Optional[str] = None

class MemoryEntry(BaseModel):
    session_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GuardrailResult(BaseModel):
    passed: bool
    modified_content: Optional[str] = None
    reason: Optional[str] = None

class AgentCard(BaseModel):
    name: str
    description: str
    role: str
    tools: List[str] = Field(default_factory=list)

class AgentContext(BaseModel):
    session_id: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReasoningStep(BaseModel):
    step_type: ReActStep
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[str] = None

class TaskRequest(BaseModel):
    source_agent: str
    query: str
    task_id: str = "default"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TaskResponse(BaseModel):
    task_id: str
    source_agent: str
    status: TaskStatus
    result: Optional[str] = None
    error: Optional[str] = None
    reasoning_trace: Optional[List[ReasoningStep]] = None
    latency_ms: Optional[float] = None
