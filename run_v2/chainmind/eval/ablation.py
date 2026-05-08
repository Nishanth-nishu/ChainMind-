"""
ChainMind Ablation Study — Table 4.

Tests four controlled ablations of the full ChainMind system to quantify
the contribution of each architectural component:

  Ablation 1: No Memory (--no-memory)
      Disables both STM and LTM. Orchestrator has no episodic recall.
      Measures: does memory improve multi-turn / context-dependent tasks?

  Ablation 2: No Tool Isolation (--no-isolation)
      All agents receive ALL MCP tools (molecular + research + KG tools).
      Measures: does per-agent tool isolation reduce reasoning errors?

  Ablation 3: No Force-Convergence (--no-convergence)
      Removes the max-steps loop from BaseAgent. Agent must self-terminate.
      Measures: how often does the ReAct loop diverge without forced stopping?

  Ablation 4: No A2A Routing (--no-a2a)
      Orchestrator calls tools directly instead of delegating to specialists.
      Measures: does the A2A delegation improve task decomposition accuracy?

Usage
-----
  # Run all 4 ablations on a 20-task sample
  python -m chainmind.eval.ablation --n 20

  # Run a specific ablation on the full benchmark
  python -m chainmind.eval.ablation --ablation no_memory --mode full

  # Generate Table 4 from saved ablation results
  python -m chainmind.eval.ablation --mode report --results-dir results/bench/
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from chainmind.config.settings import Settings
from chainmind.core.types import AgentContext, TaskRequest
from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark, score_response

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results/bench")

ABLATION_NAMES = ["full", "no_memory", "no_isolation", "no_convergence", "no_a2a"]


# ---------------------------------------------------------------------------
# Ablation-aware agent builder
# ---------------------------------------------------------------------------

def build_orchestrator(
    settings: Settings,
    no_memory: bool = False,
    no_isolation: bool = False,
    no_convergence: bool = False,
    no_a2a: bool = False,
):
    """Construct an OrchestratorAgent with the specified ablation flags."""
    from chainmind.llm.router import LLMRouter
    from chainmind.agents.orchestrator import OrchestratorAgent
    from chainmind.agents.specialists import (
        ComputationalChemistAgent,
        WebResearchAgent,
        KnowledgeGraphAgent,
    )
    from chainmind.a2a.protocol import AgentRegistry
    from chainmind.mcp.molecular_server import MolecularMCPServer
    from chainmind.mcp.research_server import ResearchMCPServer

    router = LLMRouter(settings)
    mol_server = MolecularMCPServer()
    res_server = ResearchMCPServer()

    if no_isolation:
        # Give ALL tools to ALL agents (anti-isolation)
        all_servers = [mol_server, res_server]
        chem_servers = all_servers
        web_servers = all_servers
        kg_servers = all_servers
    else:
        # Normal: each agent gets only its own MCP server
        chem_servers = [mol_server]
        web_servers = [res_server]
        kg_servers = [res_server]

    chem_agent = ComputationalChemistAgent(llm_router=router, mcp_servers=chem_servers)
    web_agent  = WebResearchAgent(llm_router=router, mcp_servers=web_servers)
    kg_agent   = KnowledgeGraphAgent(llm_router=router, mcp_servers=kg_servers)

    # Apply no_convergence: monkey-patch max_steps to a very large number
    if no_convergence:
        for agent in [chem_agent, web_agent, kg_agent]:
            agent._max_steps = 999  # effectively unlimited

    if no_a2a:
        # No A2A: orchestrator answers directly (no delegation)
        registry = None
    else:
        registry = AgentRegistry()
        registry.register(chem_agent)
        registry.register(web_agent)
        registry.register(kg_agent)

    memory_store = None
    if not no_memory:
        try:
            from chainmind.memory.manager import MemoryManager
            memory_store = MemoryManager()
        except Exception as e:
            logger.warning(f"Memory init failed (non-fatal): {e}")

    return OrchestratorAgent(
        llm_router=router,
        agent_registry=registry,
        memory_store=memory_store,
    )


# ---------------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------------

async def run_ablation(
    ablation_name: str,
    tasks: list[dict],
    settings: Settings,
) -> list[dict]:
    """Run one ablation configuration on all tasks. Returns per-task results."""
    flags = {
        "no_memory":      ablation_name == "no_memory",
        "no_isolation":   ablation_name == "no_isolation",
        "no_convergence": ablation_name == "no_convergence",
        "no_a2a":         ablation_name == "no_a2a",
    }
    orch = build_orchestrator(settings, **flags)

    results = []
    for i, task in enumerate(tasks):
        print(f"  [{i+1:3d}/{len(tasks)}] {ablation_name:<16} | {task['id']:6s} | ", end="", flush=True)
        start = time.perf_counter()
        try:
            task_req = TaskRequest(source_agent="ablation_runner", query=task["query"])
            ctx = AgentContext(session_id=str(uuid.uuid4()))
            resp = await orch.process(task_req, ctx)
            response_text = resp.result or resp.error or ""
        except Exception as e:
            response_text = f"ERROR: {e}"

        latency_ms = (time.perf_counter() - start) * 1000
        score_result = score_response(task, response_text)
        icon = "✅" if score_result["passed"] else "❌"
        print(f"{icon} score={score_result['score']:.2f}")

        results.append({
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task.get("difficulty", "medium"),
            "ablation": ablation_name,
            "score": score_result["score"],
            "passed": score_result["passed"],
            "latency_ms": round(latency_ms, 1),
        })

    return results


# ---------------------------------------------------------------------------
# Table 4 computation
# ---------------------------------------------------------------------------

def compute_table4(all_results: list[dict]) -> dict[str, Any]:
    """
    Compute Table 4: Ablation Study.

    For each ablation, report: overall TSR, per-category TSR, avg latency.
    The delta columns (vs Full) are computed automatically.
    """
    ablations = ["full", "no_memory", "no_isolation", "no_convergence", "no_a2a"]
    cats = ["A", "B", "C", "D"]
    rows: dict[str, dict] = {}

    for ablation in ablations:
        abl_results = [r for r in all_results if r["ablation"] == ablation]
        if not abl_results:
            continue

        row: dict[str, Any] = {}
        # Overall TSR
        row["tsr_overall"] = round(
            sum(1 for r in abl_results if r["passed"]) / max(len(abl_results), 1) * 100, 1
        )
        # Per-category TSR
        for cat in cats:
            cat_r = [r for r in abl_results if r["category"] == cat]
            if cat_r:
                row[f"tsr_cat_{cat}"] = round(
                    sum(1 for r in cat_r if r["passed"]) / len(cat_r) * 100, 1
                )
        # Avg latency
        row["avg_latency_ms"] = round(
            sum(r["latency_ms"] for r in abl_results) / max(len(abl_results), 1), 1
        )
        rows[ablation] = row

    # Compute delta vs full
    if "full" in rows:
        full_tsr = rows["full"]["tsr_overall"]
        for ablation in ablations:
            if ablation != "full" and ablation in rows:
                rows[ablation]["delta_tsr"] = round(rows[ablation]["tsr_overall"] - full_tsr, 1)

    return {
        "columns": ["Overall TSR%", "Cat-A", "Cat-B", "Cat-C", "Cat-D", "Δ vs Full", "Avg Lat(ms)"],
        "rows": rows,
    }


def print_table4(table4: dict) -> None:
    """Pretty-print Table 4 to console."""
    print("\n" + "="*80)
    print("TABLE 4: Ablation Study — Component Contribution")
    print("="*80)
    print(f"  {'Ablation':<22} {'Overall':>8} {'Cat-A':>6} {'Cat-B':>6} {'Cat-C':>6} "
          f"{'Cat-D':>6} {'Δ Full':>7} {'Lat(ms)':>8}")
    print("-"*80)
    LABEL = {
        "full":           "Full System",
        "no_memory":      "– No Memory",
        "no_isolation":   "– No Isolation",
        "no_convergence": "– No Convergence",
        "no_a2a":         "– No A2A",
    }
    for abl, row in table4["rows"].items():
        label = LABEL.get(abl, abl)
        delta = f"{row.get('delta_tsr', 0.0):+.1f}" if "delta_tsr" in row else "  ref"
        print(
            f"  {label:<22} "
            f"{row.get('tsr_overall', 'N/A'):>8} "
            f"{row.get('tsr_cat_A', '-'):>6} "
            f"{row.get('tsr_cat_B', '-'):>6} "
            f"{row.get('tsr_cat_C', '-'):>6} "
            f"{row.get('tsr_cat_D', '-'):>6} "
            f"{delta:>7} "
            f"{row.get('avg_latency_ms', '-'):>8}"
        )


def generate_table4_markdown(table4: dict) -> str:
    """Render Table 4 as Markdown."""
    lines = [
        "## Table 4: Ablation Study",
        "",
        "| Configuration | Overall TSR% | Cat-A | Cat-B | Cat-C | Cat-D | Δ vs Full | Avg Lat (ms) |",
        "|--------------|-------------:|------:|------:|------:|------:|----------:|-------------:|",
    ]
    LABEL = {
        "full": "**Full System** *(baseline)*",
        "no_memory": "– No Memory (STM/LTM disabled)",
        "no_isolation": "– No Tool Isolation",
        "no_convergence": "– No Force-Convergence",
        "no_a2a": "– No A2A Protocol",
    }
    for abl, row in table4["rows"].items():
        label = LABEL.get(abl, abl)
        delta = f"{row.get('delta_tsr', 0.0):+.1f}" if "delta_tsr" in row else "ref"
        lines.append(
            f"| {label} "
            f"| {row.get('tsr_overall', '—')} "
            f"| {row.get('tsr_cat_A', '—')} "
            f"| {row.get('tsr_cat_B', '—')} "
            f"| {row.get('tsr_cat_C', '—')} "
            f"| {row.get('tsr_cat_D', '—')} "
            f"| {delta} "
            f"| {row.get('avg_latency_ms', '—')} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ChainMind Ablation Study (Table 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ablation",
                        choices=ABLATION_NAMES + ["all"],
                        default="all",
                        help="Which ablation to run (default: all)")
    parser.add_argument("--mode", choices=["sample", "full", "report"],
                        default="sample", help="Task selection mode")
    parser.add_argument("--n", type=int, default=20,
                        help="Tasks per ablation in 'sample' mode")
    parser.add_argument("--category", choices=["A", "B", "C", "D", "all"], default="all")
    parser.add_argument("--output-dir", default="results/bench")
    parser.add_argument("--results-dir", default=None,
                        help="Load saved ablation results for 'report' mode")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "report":
        # Load all bench_*.json files in results-dir and regenerate Table 4
        results_dir = Path(args.results_dir or args.output_dir)
        all_results: list[dict] = []
        for f in sorted(results_dir.glob("ablation_*.json")):
            with open(f) as fp:
                data = json.load(fp)
            all_results.extend(data.get("results", []))
        if not all_results:
            print(f"No ablation results found in {results_dir}")
            return
        table4 = compute_table4(all_results)
        print_table4(table4)
        md = generate_table4_markdown(table4)
        md_path = out_dir / "table4_ablation.md"
        md_path.write_text(md)
        print(f"\n  📄 Table 4 Markdown → {md_path}")
        return

    # Load tasks
    all_tasks = load_benchmark()
    if args.category != "all":
        all_tasks = [t for t in all_tasks if t["category"] == args.category]

    tasks = all_tasks[:args.n] if args.mode == "sample" else all_tasks

    ablations_to_run = ABLATION_NAMES if args.ablation == "all" else [args.ablation]

    settings = Settings()
    all_results: list[dict] = []

    print(f"\n  ChainMind Ablation Study")
    print(f"  Ablations : {ablations_to_run}")
    print(f"  Tasks     : {len(tasks)} per ablation")

    for ablation in ablations_to_run:
        print(f"\n{'='*60}")
        print(f"  Running: {ablation}")
        print(f"{'='*60}")
        results = asyncio.run(run_ablation(ablation, tasks, settings))
        all_results.extend(results)

    # Save raw results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ablation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "ablations": ablations_to_run,
                   "task_count": len(tasks), "results": all_results}, f, indent=2)
    print(f"\n  💾 Raw results → {out_path}")

    # Generate and print Table 4
    table4 = compute_table4(all_results)
    print_table4(table4)

    md = generate_table4_markdown(table4)
    md_path = out_dir / f"table4_{ts}.md"
    md_path.write_text(md)
    print(f"  📄 Table 4 Markdown → {md_path}")


if __name__ == "__main__":
    main()
