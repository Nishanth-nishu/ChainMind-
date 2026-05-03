# ChainMind

**Standards-Based Multi-Agent Drug Discovery with Local LLMs via MCP and A2A Protocols**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Benchmark: ChainMind-Bench](https://img.shields.io/badge/Benchmark-ChainMind--Bench%20v1.0-orange)](chainmind/eval/benchmarks/README.md)

---

## Overview

ChainMind is the **first open-source multi-agent drug discovery system** that combines:

1. **MCP (Model Context Protocol)** — structured tool access for molecular and research tools
2. **A2A (Agent-to-Agent) Protocol** — explicit inter-agent delegation with a typed registry
3. **Local 7B LLM (Qwen2.5-7B via vLLM)** — zero data leakage, $0/query cost

Every published competitor (ChemCrow, DrugAgent, MADD, Prompt-to-Pill) uses GPT-4 or
proprietary APIs. None use MCP/A2A. ChainMind closes that gap with a fully open stack.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User Query                           │
└──────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   OrchestratorAgent        │  ReAct loop + STM/LTM memory
         │   (Task decomposition)     │  Force-convergence (max_steps)
         └──────┬──────────┬──────────┘
                │  A2A     │  A2A
    ┌───────────▼──┐  ┌────▼────────────┐  ┌──────────────────┐
    │ Computational │  │  Web Research   │  │  Knowledge Graph │
    │ Chemist Agent │  │  Agent          │  │  Agent           │
    └───────┬───────┘  └────┬────────────┘  └────────┬─────────┘
            │ MCP           │ MCP                    │ MCP
    ┌───────▼───────┐  ┌────▼────────────┐  ┌────────▼─────────┐
    │ MolecularMCP  │  │  ResearchMCP    │  │  ResearchMCP     │
    │ Server        │  │  Server         │  │  Server          │
    │ (RDKit +      │  │  (ArXiv +       │  │  (generate_kg +  │
    │  BioPython)   │  │   DuckDuckGo +  │  │   search_lit)    │
    └───────────────┘  │   TDC)          │  └──────────────────┘
                       └─────────────────┘
```

**Tool isolation is enforced at registration time** — each agent receives only its own
MCP server. Cross-agent tool contamination is a controlled variable in the ablation study.

---

## Installation

```bash
# 1. Clone and create virtual environment
git clone https://github.com/Nishanth-nishu/ChainMind-.git
cd ChainMind-
python -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env: set VLLM_BASE_URL or OPENAI_API_KEY as needed

# 4. (Optional) Start local vLLM server
bash scripts/start_vllm_optimized.sh
```

---

## Quick Start

```bash
# Single drug discovery query via CLI
python cli.py "Does Aspirin pass Lipinski's Rule of 5?"

# Start the REST API server
python -m chainmind.api.app

# POST request
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Calculate Tanimoto similarity between Aspirin and Ibuprofen"}'
```

---

## ChainMind-Bench Evaluation

ChainMind-Bench is a **100-task deterministic benchmark** for agentic drug discovery systems.

| Category | Tasks | Scorer | Example |
|----------|------:|--------|---------|
| A — Molecular Property | 40 | Numerical / Boolean | "Does Aspirin pass Lipinski RO5?" |
| B — Literature Retrieval | 30 | Keyword recall ≥ 0.6 | "Find ArXiv papers on GNN ADMET" |
| C — Knowledge Graph | 15 | Mermaid validity + ≥ 5 edges | "KG for PROTAC mechanism" |
| D — Multi-Step Chain | 15 | Composite (A+B+C) | "Search PubChem then assess drug-likeness" |

### Running the benchmark

```bash
# ── Smoke test (5 tasks, local Qwen system) ──────────────────────────────────
python -m chainmind.eval.bench_runner --mode sample --n 5 --system chainmind_qwen

# ── Full ChainMind-Bench, all baselines (requires vLLM running) ──────────────
python -m chainmind.eval.bench_runner --mode full --system all --output-dir results/

# ── Via unified runner ───────────────────────────────────────────────────────
python -m chainmind.eval.runner --mode bench --bench-system chainmind_qwen --bench-n 10
python -m chainmind.eval.runner --mode bench-full --bench-system all

# ── Ablation study (Table 4) ─────────────────────────────────────────────────
python -m chainmind.eval.ablation --n 20
python -m chainmind.eval.ablation --mode full

# ── Re-render tables from saved JSON ─────────────────────────────────────────
python -m chainmind.eval.bench_runner --mode report --results-file results/bench/bench_full_*.json
```

### Expected output tables

The harness produces **5 papers-ready tables**:

```
TABLE 1: Task Success Rate (TSR %) — Higher is better
────────────────────────────────────────────────────────────────────────
System                            Cat-A  Cat-B  Cat-C  Cat-D   Avg
────────────────────────────────────────────────────────────────────────
  Qwen-7B (no tools)               ...    ...    ...    ...    ...
  ChainMind (Qwen-7B)              ...    ...    ...    ...    ...
  ReAct-only (Qwen-7B)             ...    ...    ...    ...    ...
  GPT-4o (no tools)                ...    ...    ...    ...    ...
  ChainMind (GPT-4o)               ...    ...    ...    ...    ...

TABLE 4: Ablation Study — Component Contribution
────────────────────────────────────────────────────────────────────────
Ablation               Overall  Cat-A  Cat-B  Cat-C  Cat-D  Δ Full  Lat(ms)
────────────────────────────────────────────────────────────────────────
  Full System            ...      ...    ...    ...    ...    ref     ...
  – No Memory            ...      ...    ...    ...    ...    ...     ...
  – No Tool Isolation    ...      ...    ...    ...    ...    ...     ...
  – No Force-Converge    ...      ...    ...    ...    ...    ...     ...
  – No A2A               ...      ...    ...    ...    ...    ...     ...
```

---

## Running Tests

```bash
# All tests (after cleanup, zero SC tests remain)
python -m pytest tests/ -x -q

# D4 unit tests only
python -m pytest tests/ -k "d4 or molecular or chemistry or research" -v

# Benchmark validator self-check
python -c "
from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark, validate_all
tasks = load_benchmark()
print(f'Loaded {len(tasks)} tasks')
validate_all()
"
```

---

## Module Map (D4-relevant files)

```
chainmind/
├── agents/
│   ├── base_agent.py       # ReAct loop + force-convergence + circuit breaker
│   ├── orchestrator.py     # Task decomposition + A2A delegation + STM/LTM
│   └── specialists.py      # 3 D4 specialists (Chem, Web, KG)
├── mcp/
│   ├── molecular_server.py # RDKit + BioPython MCP server (Lipinski, Tanimoto, 3D)
│   └── research_server.py  # ArXiv + DuckDuckGo + TDC MCP server
├── a2a/
│   └── protocol.py         # Agent registry + typed A2A routing
├── memory/
│   ├── stm.py              # Short-Term Memory (session-scoped ring buffer)
│   ├── ltm.py              # Long-Term Memory (ChromaDB vector store)
│   └── manager.py          # Unified STM/LTM manager
├── llm/
│   └── router.py           # vLLM / OpenAI provider with circuit breaker + fallback
├── config/
│   └── constants.py        # AgentRole, ToolCategory, ReActStep enums (D4-only)
└── eval/
    ├── dataset.py           # 15 D4 EvalQuestion objects (legacy runner compat)
    ├── bench_runner.py      # 5-baseline evaluation harness → Tables 1–3, 5
    ├── ablation.py          # 4-ablation study → Table 4
    └── benchmarks/
        ├── chainmind_bench.json       # 100-task ground-truth dataset
        ├── ground_truth_validator.py  # Automated scoring (numerical / KW / Mermaid)
        └── README.md                  # Dataset card
```

---

## Research Contributions

This system targets **NeurIPS 2026 AI4Science Workshop** and makes three claims:

1. **Architecture**: First open-source multi-agent D4 system built on interoperable
   MCP (tool access) and A2A (agent coordination) standards — enables modular agent
   composition without vendor lock-in.

2. **Efficiency**: A local 7B parameter model with forced convergence and MCP tool
   isolation achieves competitive accuracy on drug discovery tasks at **$0/query** vs
   ~$0.03–0.15/query for GPT-4-based systems.

3. **Benchmark**: ChainMind-Bench — 100 tasks with deterministic ground-truth answers
   spanning Lipinski analysis, molecular similarity, literature retrieval, KG generation,
   and multi-step chains. Fully reproducible from public APIs and RDKit.

---

## Citation

```bibtex
@misc{chainmind2026,
  title        = {ChainMind: Standards-Based Multi-Agent Drug Discovery
                  with Local LLMs via MCP and A2A Protocols},
  author       = {Nishanth R.},
  year         = {2026},
  howpublished = {\url{https://github.com/Nishanth-nishu/ChainMind-}},
  note         = {Preprint. Target: NeurIPS 2026 AI4Science Workshop.}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
