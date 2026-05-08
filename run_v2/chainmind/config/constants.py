"""
ChainMind Constants — Enums and constant values for the D4 platform.

All magic numbers and strings are centralized here to avoid scattered
literal values throughout the codebase. Keeping these in one place
also simplifies ablation experiments (e.g. toggling circuit-breaker
thresholds without hunting through agent code).
"""

from enum import Enum, auto


class AgentRole(str, Enum):
    """
    Agent specialization roles for the D4 platform.

    Only roles with active specialist implementations are enumerated here.
    Adding a role without a corresponding agent class will raise an
    AgentNotFoundError at routing time.
    """
    ORCHESTRATOR = "orchestrator"
    COMPUTATIONAL_CHEMISTRY = "computational_chemistry"
    WEB_RESEARCH = "web_research"
    KNOWLEDGE_GRAPH = "knowledge_graph"

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
    """
    MCP tool categories for the D4 platform.

    Used by the MCP servers to classify tools; also used by the benchmark
    runner to report tool-usage statistics per category.
    """
    MOLECULAR = "molecular"        # RDKit / BioPython chemistry tools
    KNOWLEDGE_BASE = "knowledge_base"
    ANALYTICS = "analytics"
    EXTERNAL_API = "external_api"
    ANALYSIS = "analysis"          # General computational analysis
    SEARCH = "search"              # Web / ArXiv / TDC search


# --- Defaults ---
DEFAULT_SYSTEM_PROMPT_VERSION = "v1.0"
MAX_RETRIEVAL_CANDIDATES = 50
RRF_K_CONSTANT = 60  # Standard RRF constant from the literature
CIRCUIT_BREAKER_MIN_CALLS = 3  # Minimum calls before circuit can trip
