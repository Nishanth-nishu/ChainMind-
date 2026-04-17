"""
ChainMind Constants — Enums and constant values.

All magic numbers and strings are centralized here.
"""

from enum import Enum, auto


class AgentRole(str, Enum):
    """Agent specialization roles."""
    ORCHESTRATOR = "orchestrator"
    DEMAND_FORECASTING = "demand_forecasting"
    INVENTORY_MANAGEMENT = "inventory_management"
    PROCUREMENT = "procurement"
    LOGISTICS = "logistics"
    QUALITY_ASSURANCE = "quality_assurance"


class TaskStatus(str, Enum):
    """A2A task lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — stop traffic
    HALF_OPEN = "half_open"  # Testing recovery


class ReActStep(str, Enum):
    """ReAct reasoning loop stages."""
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    VERIFY = "verify"
    REFLECT = "reflect"


class GuardrailAction(str, Enum):
    """Actions a guardrail can take."""
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    MODIFY = "modify"


class ToolCategory(str, Enum):
    """MCP tool categories."""
    SUPPLY_CHAIN = "supply_chain"
    KNOWLEDGE_BASE = "knowledge_base"
    ANALYTICS = "analytics"
    EXTERNAL_API = "external_api"


# --- Defaults ---
DEFAULT_SYSTEM_PROMPT_VERSION = "v1.0"
MAX_RETRIEVAL_CANDIDATES = 50
RRF_K_CONSTANT = 60  # Standard RRF constant from the literature
CIRCUIT_BREAKER_MIN_CALLS = 3  # Minimum calls before circuit can trip
