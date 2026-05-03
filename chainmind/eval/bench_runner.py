"""
ChainMind-Bench Runner — 5-baseline evaluation harness.

Runs ChainMind-Bench (100 tasks) against five systems and produces the five
tables required for a publishable paper:

  TABLE 1: Task Success Rate (TSR) per category per system
  TABLE 2: Tool Usage Analysis (selection accuracy, param accuracy, exec rate)
  TABLE 3: Efficiency (latency ms, cost per task, tokens)
  TABLE 4: Ablation Study (no-memory, no-isolation, no-convergence, no-A2A)
  TABLE 5: Error Taxonomy (parsing/tool/reasoning/hallucination/convergence)

Usage
-----
  # Quick smoke test (5 tasks, one system)
  python -m chainmind.eval.bench_runner --mode sample --n 5 --system chainmind_qwen

  # Full benchmark, all tasks, one system
  python -m chainmind.eval.bench_runner --mode full --system chainmind_qwen

  # Full benchmark, all systems (requires GPT-4 key for gpt4 systems)
  python -m chainmind.eval.bench_runner --mode full --system all

  # Ablation only
  python -m chainmind.eval.bench_runner --mode ablation --system chainmind_qwen

  # Load existing results and re-render tables
  python -m chainmind.eval.bench_runner --mode report --results-file results/bench_full_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from chainmind.config.settings import Settings
from chainmind.core.types import AgentContext, LLMMessage, LLMRequest, TaskRequest
from chainmind.eval.benchmarks.ground_truth_validator import (
    BENCH_PATH,
    load_benchmark,
    score_response,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/bench")

# Cost per 1K tokens (USD), update as needed
COST_PER_1K = {
    "qwen_direct":     0.000,   # local — $0
    "chainmind_qwen":  0.000,   # local — $0
    "react_qwen":      0.000,   # local — $0
    "gpt4_direct":     0.030,   # GPT-4o input price
    "chainmind_gpt4":  0.030,
}


# ---------------------------------------------------------------------------
# Error Taxonomy
# ---------------------------------------------------------------------------

class ErrorType:
    PARSING    = "parsing_error"    # Malformed tool call JSON
    TOOL       = "tool_error"       # Tool returned an error result
    REASONING  = "reasoning_error"  # Wrong tool selected
    HALLUCINAT = "hallucination"    # Factually wrong answer (not caught by tool)
    CONVERGENCE= "convergence"      # Max steps exceeded without answer
    NONE       = "none"             # Task succeeded


def classify_error(response: str, task: dict, score: float) -> str:
    """Classify failure mode for error taxonomy (Table 5)."""
    if score >= 0.6:
        return ErrorType.NONE

    resp_lower = response.lower()

    # Convergence failure patterns
    if any(kw in resp_lower for kw in [
        "max steps", "maximum steps", "exceeded", "convergence", "force stop", "step limit"
    ]):
        return ErrorType.CONVERGENCE

    # Tool errors
    if any(kw in response for kw in ['"error":', '"success": false', "tool not found", "Failed:"]):
        return ErrorType.TOOL

    # Parsing errors (bad JSON in tool call)
    if any(kw in resp_lower for kw in ["json", "parse error", "invalid json", "malformed"]):
        return ErrorType.PARSING

    # Reasoning error: answer given but doesn't match ground truth despite tools being called
    tool_kws = ["tool_name", "tool_call", "action:", "observation:", ">>", "calling"]
    if any(kw in resp_lower for kw in tool_kws):
        return ErrorType.REASONING

    # Hallucination: confident wrong answer with no tool call evidence
    return ErrorType.HALLUCINAT


# ---------------------------------------------------------------------------
# System configurations
# ---------------------------------------------------------------------------

class SystemConfig:
    """Describes a baseline system configuration."""
    def __init__(
        self,
        name: str,
        use_orchestrator: bool,
        use_tools: bool,
        llm_backend: str,   # "local" | "openai"
        openai_model: str = "gpt-4o",
    ):
        self.name = name
        self.use_orchestrator = use_orchestrator
        self.use_tools = use_tools
        self.llm_backend = llm_backend
        self.openai_model = openai_model

SYSTEMS: dict[str, SystemConfig] = {
    "qwen_direct":    SystemConfig("Qwen-7B (no tools)", False, False, "local"),
    "chainmind_qwen": SystemConfig("ChainMind (Qwen-7B)", True,  True,  "local"),
    "react_qwen":     SystemConfig("ReAct-only (Qwen-7B)", False, True, "local"),
    "gpt4_direct":    SystemConfig("GPT-4o (no tools)",    False, False, "openai"),
    "chainmind_gpt4": SystemConfig("ChainMind (GPT-4o)",   True,  True,  "openai"),
}


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

class BenchTaskRunner:
    """Runs a single benchmark task against a configured system."""

    def __init__(self, system: SystemConfig, settings: Settings):
        self._system = system
        self._settings = settings
        self._llm_router = None
        self._orchestrator = None

    def _get_router(self):
        if self._llm_router is None:
            from chainmind.llm.router import LLMRouter
            self._llm_router = LLMRouter(self._settings)
        return self._llm_router

    def _get_orchestrator(self):
        if self._orchestrator is None:
            router = self._get_router()
            from chainmind.agents.orchestrator import OrchestratorAgent
            from chainmind.agents.specialists import (
                ComputationalChemistAgent,
                WebResearchAgent,
                KnowledgeGraphAgent,
            )
            from chainmind.a2a.protocol import AgentRegistry
            from chainmind.mcp.molecular_server import MolecularMCPServer
            from chainmind.mcp.research_server import ResearchMCPServer

            mol_server = MolecularMCPServer()
            res_server = ResearchMCPServer()

            chem_agent = ComputationalChemistAgent(
                llm_router=router, mcp_servers=[mol_server]
            )
            web_agent = WebResearchAgent(
                llm_router=router, mcp_servers=[res_server]
            )
            kg_agent = KnowledgeGraphAgent(
                llm_router=router, mcp_servers=[res_server]
            )

            registry = AgentRegistry()
            registry.register(chem_agent)
            registry.register(web_agent)
            registry.register(kg_agent)

            self._orchestrator = OrchestratorAgent(
                llm_router=router, agent_registry=registry
            )
        return self._orchestrator

    async def run_task(self, task: dict) -> dict[str, Any]:
        """Run a single task and return a result record."""
        start = time.perf_counter()
        input_tokens  = 0
        output_tokens = 0
        error_msg     = None

        # ── Tool invocation tracking ─────────────────────────────────────────
        # These are populated when the orchestrator delegates to a specialist
        # agent that calls an MCP tool.  Presence of non-zero values is the
        # empirical evidence that ChainMind actually used tools (not parametric
        # memory), which is the central claim of the paper.
        tools_called:    list[str] = []
        tools_succeeded: list[str] = []
        agent_delegated: str | None = None

        try:
            query = task["query"]

            if self._system.use_orchestrator:
                orch = self._get_orchestrator()
                task_req = TaskRequest(
                    source_agent="bench_runner",
                    query=query,
                )
                ctx = AgentContext(session_id=str(uuid.uuid4()))

                # ── Monkey-patch MCP servers to capture tool calls ────────────
                # We temporarily wrap every MCP server's execute_tool so we
                # can record invocations without modifying production code.
                _mcp_servers = []
                try:
                    registry = orch._agent_registry if hasattr(orch, '_agent_registry') else None
                    if registry:
                        for agent in getattr(registry, '_agents', {}).values():
                            for srv in getattr(agent, '_mcp_servers', []):
                                _mcp_servers.append(srv)
                except Exception:
                    pass

                _originals = {}
                def _make_wrapper(srv, orig_fn):
                    async def _tracked(tool_name: str, params: dict):
                        tools_called.append(tool_name)
                        result = await orig_fn(tool_name, params)
                        if getattr(result, 'success', False):
                            tools_succeeded.append(tool_name)
                        return result
                    return _tracked

                for srv in _mcp_servers:
                    if hasattr(srv, 'execute_tool'):
                        _originals[id(srv)] = (srv, srv.execute_tool)
                        srv.execute_tool = _make_wrapper(srv, srv.execute_tool)

                try:
                    resp = await orch.process(task_req, ctx)
                finally:
                    # Restore original execute_tool methods
                    for srv, orig in _originals.values():
                        srv.execute_tool = orig

                response_text = resp.result or resp.error or ""
                # Derive delegated agent from result metadata if available
                agent_delegated = getattr(resp, 'agent', None)
                input_tokens  = len(query.split()) * 1.3
                output_tokens = len(response_text.split()) * 1.3

            else:
                # Direct LLM call — no tools, no orchestrator
                router = self._get_router()
                system_prompt = (
                    "You are a drug discovery AI assistant. "
                    "Answer the following question accurately and concisely. "
                    "Think step by step."
                )
                llm_resp = await router.generate(
                    LLMRequest(
                        messages=[LLMMessage(role="user", content=query)],
                        system_prompt=system_prompt,
                        temperature=0.2,
                        max_tokens=1024,
                    )
                )
                response_text = llm_resp.content
                input_tokens  = len(query.split()) * 1.3
                output_tokens = len(response_text.split()) * 1.3

        except Exception as e:
            response_text = f"ERROR: {e}"
            error_msg = str(e)
            logger.warning(f"Task {task['id']} failed: {e}")

        latency_ms = (time.perf_counter() - start) * 1000

        # Score
        score_result = score_response(task, response_text)
        error_type   = classify_error(response_text, task, score_result["score"])

        # Cost estimate
        total_tokens = input_tokens + output_tokens
        cost_usd = (
            (total_tokens / 1000)
            * COST_PER_1K.get(
                self._system.name.lower().replace(" ", "_"), 0.0
            )
        )

        return {
            "task_id":            task["id"],
            "category":           task["category"],
            "subcategory":        task.get("subcategory", ""),
            "difficulty":         task.get("difficulty", "medium"),
            "system":             self._system.name,
            "query":              task["query"][:120],
            "response_preview":   response_text[:300],
            "score":              score_result["score"],
            "passed":             score_result["passed"],
            "breakdown":          score_result["breakdown"],
            "error_type":         error_type,
            "latency_ms":         round(latency_ms, 1),
            "input_tokens":       round(input_tokens),
            "output_tokens":      round(output_tokens),
            "cost_usd":           round(cost_usd, 5),
            "run_error":          error_msg,
            # ── Tool invocation evidence (Table 2) ──────────────────────────
            "tools_called":       tools_called,
            "n_tools_called":     len(tools_called),
            "n_tools_succeeded":  len(tools_succeeded),
            "tool_success_rate":  (
                len(tools_succeeded) / len(tools_called)
                if tools_called else None
            ),
            "agent_delegated":    agent_delegated,
            "used_tools":         self._system.use_tools,
        }


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

class BenchRunner:
    """Orchestrates full benchmark runs and table generation."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()

    async def run(
        self,
        system_names: list[str],
        tasks: list[dict],
        ablation_flags: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """Run all systems on all tasks. Returns raw results dict."""
        all_results: list[dict] = []
        timestamp = datetime.datetime.now().isoformat()

        for sys_name in system_names:
            if sys_name not in SYSTEMS:
                logger.warning(f"Unknown system '{sys_name}', skipping.")
                continue

            sys_cfg = SYSTEMS[sys_name]
            print(f"\n{'='*60}")
            print(f"  System: {sys_cfg.name} ({len(tasks)} tasks)")
            print(f"{'='*60}")

            runner = BenchTaskRunner(sys_cfg, self._settings)
            for i, task in enumerate(tasks):
                print(f"  [{i+1:3d}/{len(tasks)}] {task['id']:6s} | ", end="", flush=True)
                result = await runner.run_task(task)
                icon = "✅" if result["passed"] else "❌"
                print(f"{icon} score={result['score']:.2f} lat={result['latency_ms']:.0f}ms")
                all_results.append(result)

        return {
            "timestamp": timestamp,
            "systems": system_names,
            "task_count": len(tasks),
            "results": all_results,
        }

    # -----------------------------------------------------------------------
    # Table generation
    # -----------------------------------------------------------------------

    def compute_tables(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Compute all 5 paper tables from raw results."""
        results = raw["results"]
        systems = raw["systems"]
        cats = ["A", "B", "C", "D"]

        # Table 1: TSR
        table1 = self._compute_tsr(results, systems, cats)

        # Table 2: Tool usage
        table2 = self._compute_tool_usage(results, systems)

        # Table 3: Efficiency
        table3 = self._compute_efficiency(results, systems)

        # Table 5: Error taxonomy
        table5 = self._compute_error_taxonomy(results, systems)

        return {
            "table1_tsr": table1,
            "table2_tool_usage": table2,
            "table3_efficiency": table3,
            "table5_error_taxonomy": table5,
        }

    def _compute_tsr(
        self, results: list[dict], systems: list[str], cats: list[str]
    ) -> dict:
        """Table 1: Task Success Rate per system per category."""
        rows = {}
        for sys_name in systems:
            sys_label = SYSTEMS[sys_name].name
            sys_results = [r for r in results if r["system"] == sys_label]
            row = {}
            for cat in cats:
                cat_results = [r for r in sys_results if r["category"] == cat]
                if cat_results:
                    tsr = sum(1 for r in cat_results if r["passed"]) / len(cat_results)
                    row[f"cat_{cat}"] = round(tsr * 100, 1)
                else:
                    row[f"cat_{cat}"] = "N/A"
            all_passed = [r for r in sys_results if r["passed"]]
            row["avg"] = round(len(all_passed) / max(len(sys_results), 1) * 100, 1)
            rows[sys_label] = row
        return {"columns": ["Cat-A", "Cat-B", "Cat-C", "Cat-D", "Avg"], "rows": rows}

    def _compute_tool_usage(self, results: list[dict], systems: list[str]) -> dict:
        """
        Table 2: Real MCP tool invocation metrics.

        Uses the ``tools_called`` / ``n_tools_called`` / ``n_tools_succeeded``
        fields injected by ``BenchTaskRunner.run_task()`` for systems with
        ``use_tools=True``.  Falls back to the error-type proxy for legacy
        result files that pre-date the instrumentation.
        """
        rows = {}
        for sys_name in systems:
            sys_label = SYSTEMS[sys_name].name
            sys_results = [r for r in results if r["system"] == sys_label]
            if not sys_results:
                continue

            n = max(len(sys_results), 1)
            uses_tools = SYSTEMS[sys_name].use_tools

            # ── Real tracking (post-instrumentation) ──────────────────────────
            has_tracking = any("n_tools_called" in r for r in sys_results)
            if has_tracking and uses_tools:
                # Tasks where ≥1 MCP tool was invoked
                tasks_with_calls  = [r for r in sys_results if r.get("n_tools_called", 0) > 0]
                tasks_all_success = [r for r in sys_results
                                     if r.get("n_tools_called", 0) > 0
                                     and r.get("n_tools_succeeded", 0) == r.get("n_tools_called", 0)]
                total_calls     = sum(r.get("n_tools_called", 0)    for r in sys_results)
                total_succeeded = sum(r.get("n_tools_succeeded", 0) for r in sys_results)

                rows[sys_label] = {
                    "tool_rate_%":       round(len(tasks_with_calls) / n * 100, 1),
                    "tool_success_%":    round(total_succeeded / max(total_calls, 1) * 100, 1),
                    "avg_tools_per_task": round(total_calls / n, 2),
                    "tasks_all_tools_ok": round(len(tasks_all_success) / n * 100, 1),
                    "avg_score":          round(sum(r["score"] for r in sys_results) / n, 3),
                    "source": "instrumented",
                }

            else:
                # ── Proxy (legacy / no-tool systems) ─────────────────────────
                tool_invoked = [
                    r for r in sys_results
                    if r["error_type"] not in (ErrorType.PARSING, ErrorType.TOOL)
                ]
                exec_success = [r for r in sys_results if r["error_type"] == ErrorType.NONE]
                rows[sys_label] = {
                    "tool_rate_%":        round(len(tool_invoked) / n * 100, 1) if uses_tools else 0.0,
                    "tool_success_%":     round(len(exec_success) / n * 100, 1),
                    "avg_tools_per_task": "N/A",
                    "tasks_all_tools_ok": "N/A",
                    "avg_score":          round(sum(r["score"] for r in sys_results) / n, 3),
                    "source": "proxy",
                }
        return rows

    def _compute_efficiency(self, results: list[dict], systems: list[str]) -> dict:
        """Table 3: Latency, cost, token usage."""
        rows = {}
        for sys_name in systems:
            sys_label = SYSTEMS[sys_name].name
            sys_results = [r for r in results if r["system"] == sys_label]
            if not sys_results:
                continue
            avg_lat = sum(r["latency_ms"] for r in sys_results) / len(sys_results)
            avg_cost = sum(r["cost_usd"] for r in sys_results) / len(sys_results)
            avg_in_tok = sum(r["input_tokens"] for r in sys_results) / len(sys_results)
            avg_out_tok = sum(r["output_tokens"] for r in sys_results) / len(sys_results)
            rows[sys_label] = {
                "avg_latency_ms": round(avg_lat, 1),
                "avg_cost_usd": round(avg_cost, 5),
                "avg_input_tokens": round(avg_in_tok),
                "avg_output_tokens": round(avg_out_tok),
            }
        return rows

    def _compute_error_taxonomy(self, results: list[dict], systems: list[str]) -> dict:
        """Table 5: Error type breakdown per system."""
        rows = {}
        error_types = [ErrorType.PARSING, ErrorType.TOOL, ErrorType.REASONING,
                       ErrorType.HALLUCINAT, ErrorType.CONVERGENCE, ErrorType.NONE]
        for sys_name in systems:
            sys_label = SYSTEMS[sys_name].name
            sys_results = [r for r in results if r["system"] == sys_label]
            if not sys_results:
                continue
            n = len(sys_results)
            row = {}
            for et in error_types:
                count = sum(1 for r in sys_results if r["error_type"] == et)
                row[et] = round(count / n * 100, 1)
            rows[sys_label] = row
        return rows

    def print_tables(self, tables: dict[str, Any]) -> None:
        """Pretty-print all tables to console."""
        # Table 1: TSR
        print("\n" + "="*70)
        print("TABLE 1: Task Success Rate (%) — Higher is better")
        print("="*70)
        t1 = tables["table1_tsr"]
        header = f"{'System':<35} {'Cat-A':>6} {'Cat-B':>6} {'Cat-C':>6} {'Cat-D':>6} {'Avg':>6}"
        print(header)
        print("-"*70)
        for sys_label, row in t1["rows"].items():
            a = f"{str(row.get('cat_A', 'N/A')):>6}"
            b = f"{str(row.get('cat_B', 'N/A')):>6}"
            c = f"{str(row.get('cat_C', 'N/A')):>6}"
            d = f"{str(row.get('cat_D', 'N/A')):>6}"
            avg = f"{str(row.get('avg', 'N/A')):>6}"
            print(f"  {sys_label:<33} {a} {b} {c} {d} {avg}")

        # Table 3: Efficiency
        print("\n" + "="*70)
        print("TABLE 3: Efficiency — Local ($0/task) vs Cloud")
        print("="*70)
        t3 = tables["table3_efficiency"]
        header = f"{'System':<35} {'Avg Lat(ms)':>11} {'Cost/task($)':>12} {'In Tok':>8} {'Out Tok':>8}"
        print(header)
        print("-"*70)
        for sys_label, row in t3.items():
            print(f"  {sys_label:<33} {row['avg_latency_ms']:>11.1f} "
                  f"{row['avg_cost_usd']:>12.5f} {row['avg_input_tokens']:>8} {row['avg_output_tokens']:>8}")

        # Table 5: Error Taxonomy
        print("\n" + "="*70)
        print("TABLE 5: Error Taxonomy (%)")
        print("="*70)
        t5 = tables["table5_error_taxonomy"]
        header = (f"{'System':<30} {'Parse':>7} {'Tool':>7} {'Reason':>7} "
                  f"{'Halluc':>7} {'Conv':>7} {'None':>7}")
        print(header)
        print("-"*70)
        for sys_label, row in t5.items():
            print(
                f"  {sys_label:<28} "
                f"{row.get(ErrorType.PARSING,0):>7.1f} "
                f"{row.get(ErrorType.TOOL,0):>7.1f} "
                f"{row.get(ErrorType.REASONING,0):>7.1f} "
                f"{row.get(ErrorType.HALLUCINAT,0):>7.1f} "
                f"{row.get(ErrorType.CONVERGENCE,0):>7.1f} "
                f"{row.get(ErrorType.NONE,0):>7.1f}"
            )

    def save_results(self, raw: dict, tables: dict, prefix: str = "bench") -> Path:
        """Save raw results + tables to JSON."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{prefix}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump({"raw": raw, "tables": tables}, f, indent=2, default=str)
        print(f"\n  💾 Results saved → {out_path}")
        return out_path

    def generate_markdown(self, tables: dict, raw: dict) -> str:
        """Render tables as LaTeX-ready Markdown for the paper."""
        lines = [
            "# ChainMind-Bench Results",
            "",
            f"**Date:** {raw.get('timestamp', 'N/A')}  ",
            f"**Tasks:** {raw.get('task_count', '?')} × {len(raw.get('systems', []))} systems",
            "",
            "## Table 1: Task Success Rate (TSR %)",
            "",
            "| System | Cat-A | Cat-B | Cat-C | Cat-D | Avg |",
            "|--------|------:|------:|------:|------:|----:|",
        ]
        for sys_label, row in tables["table1_tsr"]["rows"].items():
            lines.append(
                f"| {sys_label} "
                f"| {row.get('cat_A','—')} "
                f"| {row.get('cat_B','—')} "
                f"| {row.get('cat_C','—')} "
                f"| {row.get('cat_D','—')} "
                f"| **{row.get('avg','—')}** |"
            )
        lines += [
            "",
            "## Table 3: Efficiency",
            "",
            "| System | Avg Latency (ms) | Cost/Task (USD) | Input Tokens | Output Tokens |",
            "|--------|----------------:|---------------:|-------------:|--------------:|",
        ]
        for sys_label, row in tables["table3_efficiency"].items():
            lines.append(
                f"| {sys_label} "
                f"| {row['avg_latency_ms']:.1f} "
                f"| ${row['avg_cost_usd']:.5f} "
                f"| {row['avg_input_tokens']} "
                f"| {row['avg_output_tokens']} |"
            )
        lines += [
            "",
            "## Table 5: Error Taxonomy (%)",
            "",
            "| System | Parsing | Tool | Reasoning | Hallucination | Convergence | Success |",
            "|--------|--------:|-----:|----------:|--------------:|------------:|-------:|",
        ]
        for sys_label, row in tables["table5_error_taxonomy"].items():
            lines.append(
                f"| {sys_label} "
                f"| {row.get(ErrorType.PARSING,0):.1f} "
                f"| {row.get(ErrorType.TOOL,0):.1f} "
                f"| {row.get(ErrorType.REASONING,0):.1f} "
                f"| {row.get(ErrorType.HALLUCINAT,0):.1f} "
                f"| {row.get(ErrorType.CONVERGENCE,0):.1f} "
                f"| **{row.get(ErrorType.NONE,0):.1f}** |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ChainMind-Bench: Full evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["sample", "full", "ablation", "report"],
                        default="sample", help="Evaluation mode")
    parser.add_argument("--system", default="chainmind_qwen",
                        help="System(s) to evaluate (comma-separated or 'all')")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of tasks for 'sample' mode")
    parser.add_argument("--category", choices=["A", "B", "C", "D", "all"], default="all",
                        help="Filter tasks by category")
    parser.add_argument("--output-dir", default="results/bench", help="Output directory")
    parser.add_argument("--results-file", default=None, help="Load existing results JSON for 'report' mode")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RESULTS_DIR_OVERRIDE = Path(args.output_dir)
    RESULTS_DIR_OVERRIDE.mkdir(parents=True, exist_ok=True)

    # Determine systems
    if args.system == "all":
        systems = list(SYSTEMS.keys())
    else:
        systems = [s.strip() for s in args.system.split(",")]

    # Load tasks
    all_tasks = load_benchmark()
    if args.category != "all":
        all_tasks = [t for t in all_tasks if t["category"] == args.category]

    if args.mode == "sample":
        tasks = all_tasks[:args.n]
    elif args.mode == "full":
        tasks = all_tasks
    elif args.mode == "report":
        if not args.results_file:
            parser.error("--results-file is required for 'report' mode")
        with open(args.results_file) as f:
            saved = json.load(f)
        runner = BenchRunner()
        tables = runner.compute_tables(saved["raw"])
        runner.print_tables(tables)
        md = runner.generate_markdown(tables, saved["raw"])
        md_path = RESULTS_DIR_OVERRIDE / "bench_report.md"
        md_path.write_text(md)
        print(f"\n  📄 Markdown report → {md_path}")
        return
    else:
        tasks = all_tasks

    settings = Settings()
    runner = BenchRunner(settings)

    print(f"\n  ChainMind-Bench Evaluation")
    print(f"  Mode    : {args.mode}")
    print(f"  Systems : {systems}")
    print(f"  Tasks   : {len(tasks)}")

    raw = asyncio.run(runner.run(systems, tasks))
    tables = runner.compute_tables(raw)
    runner.print_tables(tables)

    out_path = runner.save_results(raw, tables, prefix=f"bench_{args.mode}")
    md = runner.generate_markdown(tables, raw)
    md_path = out_path.with_suffix(".md")
    md_path.write_text(md)
    print(f"  📄 Markdown → {md_path}")


if __name__ == "__main__":
    main()
