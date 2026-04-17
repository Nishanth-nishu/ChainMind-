"""
ChainMind Settings — Pydantic-based configuration management.

All configuration is externalized via environment variables.
Zero hardcoding. Follows the 12-Factor App methodology.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    OLLAMA = "ollama"


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment ---
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # --- Gemini ---
    gemini_api_keys: list[str] = Field(default_factory=list)
    gemini_model: str = "gemini-2.0-flash"

    # --- OpenAI ---
    openai_api_keys: list[str] = Field(default_factory=list)
    openai_model: str = "gpt-4o-mini"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- LLM Router ---
    llm_primary_provider: LLMProvider = LLMProvider.GEMINI
    llm_fallback_chain: list[LLMProvider] = Field(
        default_factory=lambda: [LLMProvider.GEMINI, LLMProvider.OPENAI, LLMProvider.OLLAMA]
    )
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 30
    llm_circuit_breaker_threshold: int = 5
    llm_circuit_breaker_recovery_seconds: int = 60

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_rate_limit_per_minute: int = 60
    api_key: str = "changeme-in-production"

    # --- gRPC Server ---
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_max_workers: int = 10

    # --- Retrieval / RAG ---
    embedding_model: str = "all-MiniLM-L6-v2"
    chromadb_persist_dir: str = "./data/chromadb"
    retrieval_top_k: int = 10
    reranker_top_k: int = 5
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID

    # --- Memory ---
    memory_short_term_max_tokens: int = 4000
    memory_long_term_db_path: str = "./data/memory.db"

    # --- Guardrails ---
    guardrails_enabled: bool = True
    max_input_length: int = 10000
    max_output_tokens: int = 4096
    max_agent_steps: int = 15

    # --- Observability ---
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    tracing_enabled: bool = True
    metrics_enabled: bool = True

    @field_validator("gemini_api_keys", "openai_api_keys", mode="before")
    @classmethod
    def parse_comma_separated_keys(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated API keys from env var string."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @field_validator("llm_fallback_chain", mode="before")
    @classmethod
    def parse_fallback_chain(cls, v: str | list) -> list[LLMProvider]:
        """Parse comma-separated provider names."""
        if isinstance(v, str):
            return [LLMProvider(p.strip()) for p in v.split(",") if p.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def chromadb_path(self) -> Path:
        return Path(self.chromadb_persist_dir)

    @property
    def memory_db_path(self) -> Path:
        return Path(self.memory_long_term_db_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings instance. Cached for performance."""
    return Settings()
