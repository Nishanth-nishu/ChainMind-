"""
ChainMind FastAPI Application — Production-ready REST API.

Application factory pattern with dependency injection,
lifespan management, and CORS/auth/rate-limit middleware.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from chainmind.config.settings import get_settings, Settings
from chainmind.observability.logger import setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# Application State (Dependency Injection Container)
# =============================================================================

class AppState:
    """Central dependency container — initialized at startup, injected into routes."""

    def __init__(self):
        self.settings: Settings | None = None
        self.llm_router = None
        self.orchestrator = None
        self.agent_registry = None
        self.a2a_bus = None
        self.knowledge_base = None
        self.memory_manager = None
        self.health_monitor = None
        self.metrics = None
        self.tracer = None


_app_state = AppState()


def get_app_state() -> AppState:
    return _app_state


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings)

    logger.info("=" * 60)
    logger.info("ChainMind — Starting up...")
    logger.info(f"  Environment: {settings.environment.value}")
    logger.info(f"  Primary LLM: {settings.llm_primary_provider.value}")
    logger.info(f"  Retrieval mode: {settings.retrieval_mode.value}")
    logger.info(f"  Guardrails: {'enabled' if settings.guardrails_enabled else 'disabled'}")
    logger.info("=" * 60)

    state = _app_state
    state.settings = settings

    try:
        # Initialize LLM Router
        from chainmind.llm.router import LLMRouter
        state.llm_router = LLMRouter(settings)
        logger.info(f"LLM Router: {state.llm_router.available_providers}")

        # Initialize Memory
        from chainmind.memory.manager import MemoryManager
        state.memory_manager = MemoryManager(settings)

        # Initialize Knowledge Base
        from chainmind.retrieval.knowledge_base import KnowledgeBase
        state.knowledge_base = KnowledgeBase(settings)

        # Initialize Guardrails
        guardrails = []
        if settings.guardrails_enabled:
            from chainmind.guardrails.input_guard import InputGuard
            from chainmind.guardrails.output_guard import OutputGuard
            from chainmind.guardrails.action_guard import ActionGuard
            guardrails = [
                InputGuard(max_input_length=settings.max_input_length),
                OutputGuard(max_output_tokens=settings.max_output_tokens),
                ActionGuard(),
            ]

        # Initialize MCP Servers
        from chainmind.mcp.supply_chain_server import SupplyChainMCPServer
        from chainmind.mcp.knowledge_base_server import KnowledgeBaseMCPServer
        from chainmind.mcp.analytics_server import AnalyticsMCPServer

        mcp_servers = [
            SupplyChainMCPServer(),
            KnowledgeBaseMCPServer(knowledge_base=state.knowledge_base),
            AnalyticsMCPServer(),
        ]

        # Initialize Specialist Agents
        from chainmind.agents.specialists import (
            DemandForecastingAgent,
            InventoryAgent,
            ProcurementAgent,
            LogisticsAgent,
            QualityAgent,
        )

        specialist_agents = [
            DemandForecastingAgent(
                llm_router=state.llm_router,
                mcp_servers=mcp_servers,
                guardrails=guardrails,
                memory_store=state.memory_manager,
                max_steps=settings.max_agent_steps,
            ),
            InventoryAgent(
                llm_router=state.llm_router,
                mcp_servers=mcp_servers,
                guardrails=guardrails,
                memory_store=state.memory_manager,
                max_steps=settings.max_agent_steps,
            ),
            ProcurementAgent(
                llm_router=state.llm_router,
                mcp_servers=mcp_servers,
                guardrails=guardrails,
                memory_store=state.memory_manager,
                max_steps=settings.max_agent_steps,
            ),
            LogisticsAgent(
                llm_router=state.llm_router,
                mcp_servers=mcp_servers,
                guardrails=guardrails,
                memory_store=state.memory_manager,
                max_steps=settings.max_agent_steps,
            ),
            QualityAgent(
                llm_router=state.llm_router,
                mcp_servers=mcp_servers,
                guardrails=guardrails,
                memory_store=state.memory_manager,
                max_steps=settings.max_agent_steps,
            ),
        ]

        # Initialize A2A Registry
        from chainmind.a2a.protocol import AgentRegistry, A2ABus
        state.agent_registry = AgentRegistry()
        for agent in specialist_agents:
            state.agent_registry.register(agent)

        state.a2a_bus = A2ABus(state.agent_registry)

        # Initialize Orchestrator
        from chainmind.agents.orchestrator import OrchestratorAgent
        state.orchestrator = OrchestratorAgent(
            llm_router=state.llm_router,
            agent_registry=state.agent_registry,
            guardrails=guardrails,
        )

        # Initialize Observability
        from chainmind.observability.metrics import get_metrics
        from chainmind.observability.tracer import init_tracer
        from chainmind.observability.health import HealthMonitor

        state.metrics = get_metrics()
        state.tracer = init_tracer(
            service_name="chainmind",
            enabled=settings.tracing_enabled,
        )

        state.health_monitor = HealthMonitor(check_interval_seconds=30)

        # Register health checks
        async def check_llm():
            result = await state.llm_router.health_check_all()
            return any(r["healthy"] for r in result.values())

        state.health_monitor.register_check("llm_providers", check_llm)
        state.health_monitor.register_check(
            "knowledge_base",
            lambda: state.knowledge_base is not None,
        )

        logger.info("ChainMind — All systems initialized ✓")

        yield

    finally:
        # Shutdown
        logger.info("ChainMind — Shutting down...")
        if state.health_monitor:
            await state.health_monitor.stop_monitoring()
        logger.info("ChainMind — Shutdown complete")


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ChainMind — Agentic AI for Supply Chain",
        description=(
            "Production-grade multi-agent supply chain intelligence platform "
            "with MCP, A2A, hybrid RAG, and self-healing fault tolerance."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === Routes ===

    @app.post("/api/v1/query", tags=["Agents"])
    async def process_query(request: dict[str, Any]):
        """Submit a query to the multi-agent system."""
        from chainmind.api.schemas.requests import QueryRequest, QueryResponse
        from chainmind.core.types import TaskRequest, AgentContext

        state = get_app_state()

        try:
            req = QueryRequest(**request)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        session_id = req.session_id or str(uuid.uuid4())

        # Create task request
        task = TaskRequest(
            source_agent="api_gateway",
            target_agent=req.target_agent,
            query=req.query,
            context=req.context,
        )

        # Create agent context
        context = AgentContext(session_id=session_id)

        # Track metrics
        if state.metrics:
            state.metrics.increment("api_requests_total", labels={"endpoint": "/query"})

        # Process through orchestrator
        try:
            response = await state.orchestrator.process(task, context)

            if state.metrics:
                state.metrics.observe(
                    "query_latency_ms", response.latency_ms, labels={"status": response.status.value}
                )

            return QueryResponse(
                task_id=response.task_id,
                status=response.status.value,
                result=response.result,
                error=response.error,
                source_agent=response.source_agent,
                latency_ms=response.latency_ms,
                reasoning_steps=len(response.reasoning_trace),
                reasoning_trace=[
                    {
                        "step_type": step.step_type.value,
                        "content": step.content[:500],
                        "tool_name": step.tool_name,
                    }
                    for step in response.reasoning_trace
                ],
            ).model_dump()

        except Exception as e:
            logger.error(f"Query processing error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/agents", tags=["Agents"])
    async def list_agents():
        """List all registered agents."""
        state = get_app_state()
        if not state.agent_registry:
            return {"agents": [], "total": 0}

        agents = state.agent_registry.list_all()
        return {
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "role": a.role.value,
                    "capabilities": a.capabilities,
                    "tools": a.tools,
                }
                for a in agents
            ],
            "total": len(agents),
        }

    @app.post("/api/v1/knowledge/ingest", tags=["Knowledge Base"])
    async def ingest_document(request: dict[str, Any]):
        """Ingest a document into the knowledge base."""
        from chainmind.api.schemas.requests import KnowledgeIngestRequest

        state = get_app_state()
        if not state.knowledge_base:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")

        try:
            req = KnowledgeIngestRequest(**request)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        doc_id = await state.knowledge_base.ingest(
            content=req.content,
            metadata={"title": req.title, "category": req.category, **req.metadata},
        )

        return {"doc_id": doc_id, "status": "ingested", "title": req.title}

    @app.post("/api/v1/knowledge/search", tags=["Knowledge Base"])
    async def search_knowledge(request: dict[str, Any]):
        """Search the knowledge base."""
        from chainmind.api.schemas.requests import KnowledgeSearchRequest

        state = get_app_state()
        if not state.knowledge_base:
            raise HTTPException(status_code=503, detail="Knowledge base not initialized")

        try:
            req = KnowledgeSearchRequest(**request)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        results = await state.knowledge_base.search(
            query=req.query, top_k=req.top_k, mode=req.mode
        )

        return {
            "query": req.query,
            "mode": req.mode,
            "results": [
                {
                    "doc_id": r.doc_id,
                    "content": r.content[:500],
                    "score": round(r.score, 4),
                    "retriever": r.retriever,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "total_results": len(results),
        }

    @app.get("/api/v1/health", tags=["System"])
    async def health_check():
        """System health check (K8s liveness/readiness compatible)."""
        state = get_app_state()

        # Quick provider health
        provider_status = {}
        if state.llm_router:
            try:
                provider_status = await state.llm_router.health_check_all()
            except Exception as e:
                provider_status = {"error": str(e)}

        health_report = {}
        if state.health_monitor:
            health_report = state.health_monitor.get_status_report()

        return {
            "status": health_report.get("status", "unknown"),
            "is_healthy": health_report.get("is_healthy", True),
            "is_ready": health_report.get("is_ready", True),
            "components": health_report.get("components", {}),
            "providers": provider_status,
        }

    @app.get("/api/v1/metrics", tags=["System"])
    async def get_metrics():
        """Get system metrics (Prometheus-compatible)."""
        state = get_app_state()
        if state.metrics:
            return state.metrics.export_dict()
        return {"counters": {}, "gauges": {}, "histograms": {}}

    @app.get("/api/v1/metrics/prometheus", tags=["System"])
    async def get_prometheus_metrics():
        """Get metrics in Prometheus text format."""
        state = get_app_state()
        if state.metrics:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(content=state.metrics.export_prometheus())
        return PlainTextResponse(content="")

    @app.websocket("/api/v1/stream")
    async def websocket_stream(websocket: WebSocket):
        """WebSocket endpoint for streaming agent responses."""
        await websocket.accept()
        state = get_app_state()

        try:
            while True:
                data = await websocket.receive_json()
                query = data.get("query", "")
                session_id = data.get("session_id", str(uuid.uuid4()))

                from chainmind.core.types import TaskRequest, AgentContext, LLMMessage, LLMRequest

                # Stream response tokens
                try:
                    request = LLMRequest(
                        messages=[LLMMessage(role="user", content=query)],
                        system_prompt="You are a supply chain AI assistant. Be helpful and concise.",
                        temperature=0.7,
                    )

                    async for token in state.llm_router.stream(request):
                        await websocket.send_json({"type": "token", "content": token})

                    await websocket.send_json({"type": "done"})

                except Exception as e:
                    await websocket.send_json({"type": "error", "content": str(e)})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")

    return app


# Create the app instance
app = create_app()


def main() -> None:
    """Entry point for the API server."""
    settings = get_settings()
    uvicorn.run(
        "chainmind.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,  # Single worker for dev — use gunicorn in production
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
