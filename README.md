# ChainMind — Production-Grade Agentic AI for Supply Chain

> Multi-agent supply chain intelligence platform with MCP, A2A, gRPC, hybrid RAG, and self-healing fault tolerance.

## 🏗️ Architecture

```
┌─ Client Layer ─────────────────────────────────────┐
│  FastAPI REST (8000)  │  gRPC (50051)  │  WebSocket│
└───────────────────────┬────────────────────────────┘
┌─ A2A Orchestration ───┴────────────────────────────┐
│  Orchestrator Agent → Agent Registry → A2A Bus      │
└───────────────────────┬────────────────────────────┘
┌─ Specialist Agents ───┴────────────────────────────┐
│  Demand │ Inventory │ Procurement │ Logistics │ QA  │
│         (ReAct Loop: Think→Act→Observe→Verify)      │
└───────────────────────┬────────────────────────────┘
┌─ MCP Tool Layer ──────┴────────────────────────────┐
│  Supply Chain DB │ Knowledge Base │ Analytics │ APIs│
└───────────────────────┬────────────────────────────┘
┌─ Knowledge & RAG ─────┴────────────────────────────┐
│  BM25 (vectorless) + Dense (ChromaDB) → RRF Fusion  │
│  → Cross-Encoder Reranking                           │
└───────────────────────┬────────────────────────────┘
┌─ LLM Abstraction ─────┴───────────────────────────┐
│  Router (Round-Robin + Failover + Circuit Breaker)  │
│  Gemini │ OpenAI │ Ollama (local)                   │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# 1. Clone and install
cp .env.example .env
# Edit .env with your Gemini API keys

pip install -e ".[dev]"

# 2. Seed the knowledge base
python scripts/seed_knowledge_base.py

# 3. Start the API server
make dev

# 4. Test
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the inventory levels for SKU-001?"}'
```

## 📋 Key Features

| Feature | Implementation |
|---|---|
| **Multi-Agent Orchestration** | Orchestrator + 5 specialist agents with A2A protocol |
| **ReAct Pattern** | Think→Act→Observe→Verify→Reflect reasoning loop |
| **MCP Protocol** | 3 MCP servers (Supply Chain, Knowledge Base, Analytics) |
| **Hybrid RAG** | BM25 + Dense + RRF fusion + Cross-Encoder reranking |
| **Multi-LLM** | Gemini / OpenAI / Ollama with round-robin key rotation |
| **Self-Healing** | Circuit breakers, automatic failover, health monitoring |
| **Guardrails** | Input/output/action validation, PII redaction, rate limiting |
| **Memory** | Short-term (in-memory) + Long-term (SQLite) |
| **Observability** | Structured logging, distributed tracing, Prometheus metrics |
| **gRPC** | High-performance RPC with health checks and reflection |
| **Deployment** | Docker, Docker Compose, Kubernetes (HPA) |
| **SOLID** | Interface segregation, dependency inversion, zero hardcoding |

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/query` | POST | Submit a query to the agentic system |
| `/api/v1/stream` | WS | WebSocket for streaming responses |
| `/api/v1/agents` | GET | List registered agents |
| `/api/v1/knowledge/ingest` | POST | Ingest documents into RAG |
| `/api/v1/knowledge/search` | POST | Search knowledge base |
| `/api/v1/health` | GET | K8s health probe |
| `/api/v1/metrics` | GET | System metrics |
| `/api/v1/metrics/prometheus` | GET | Prometheus format |

## 🧪 Testing

```bash
make test-unit          # Unit tests (circuit breaker, retriever, guardrails, A2A)
make test-integration   # Integration tests
make test-evals         # Evaluation pipeline
make lint               # Linting
```

## 🐳 Docker

```bash
make docker-build       # Build image
make docker-up          # Start all services
make docker-down        # Stop services
```

## 📚 Research Foundations

- **Amazon Nova Act**: Deterministic governance, constrained decoding
- **ReAct** (Yao et al.): Interleaved reasoning and acting
- **Reflexion** (Shinn et al.): Self-correcting agents
- **MCP** (Anthropic): Agent-to-tool connectivity
- **A2A** (Google): Agent-to-agent communication
- **RRF** (Cormack et al.): Score-agnostic rank fusion
- **SPLADE**: Learned sparse retrieval
