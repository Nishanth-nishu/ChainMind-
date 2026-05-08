"""
ChainMind Evaluation Runner — Unified CLI for all evaluation modes.

Usage
-----
  # ChainMind-Bench (recommended for paper evaluation)
  python -m chainmind.eval.runner --mode bench          # Sample 10 tasks, chainmind_qwen
  python -m chainmind.eval.runner --mode bench-full     # Full 100 tasks, all systems

  # Legacy quality evaluation (15 D4 EvalQuestion objects)
  python -m chainmind.eval.runner --mode quick          # 6-question smoke test
  python -m chainmind.eval.runner --mode quality        # Full 15 D4 questions

  # Infrastructure benchmarks
  python -m chainmind.eval.runner --mode performance    # vLLM latency/throughput
  python -m chainmind.eval.runner --mode rag            # RAG faithfulness metrics
  python -m chainmind.eval.runner --mode ab             # Prompt A/B testing
  python -m chainmind.eval.runner --mode full           # All of the above
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
from typing import Any

from chainmind.config.settings import Settings
from chainmind.core.types import (
    AgentContext,
    LLMMessage,
    LLMRequest,
    TaskRequest,
)
from chainmind.eval.dataset import (
    ALL_QUESTIONS,
    QUICK_EVAL_QUESTIONS,
    EvalQuestion,
    load_bench_questions,
)
from chainmind.eval.metrics import (
    EvalResult,
    EvalSummary,
    aggregate_results,
    compute_keyword_score,
    compute_latency_score,
    compute_weighted_score,
    llm_judge_evaluate,
)

logger = logging.getLogger(__name__)


class EvalRunner:
    """Unified evaluation runner for all modes."""

    def __init__(
        self,
        settings: Settings | None = None,
        output_dir: str = "results",
    ):
        self._settings = settings or Settings()
        self._output_dir = output_dir
        self._llm_router = None
        self._orchestrator = None

    def _init_llm_router(self):
        """Lazy-init LLM router."""
        if self._llm_router is None:
            from chainmind.llm.router import LLMRouter
            self._llm_router = LLMRouter(self._settings)
            print(f"  Providers: {self._llm_router.available_providers}")
        return self._llm_router

    def _init_orchestrator(self):
        """Lazy-init orchestrator with agent registry."""
        if self._orchestrator is None:
            router = self._init_llm_router()
            from chainmind.agents.orchestrator import OrchestratorAgent
            self._orchestrator = OrchestratorAgent(llm_router=router)
        return self._orchestrator

    # =========================================================================
    # Quality evaluation
    # =========================================================================

    async def run_quality_eval(
        self,
        questions: list[EvalQuestion],
        use_orchestrator: bool = False,
    ) -> EvalSummary:
        """Run quality evaluation on a set of questions."""
        router = self._init_llm_router()
        results: list[EvalResult] = []

        print(f"\n{'=' * 60}")
        print(f"  ChainMind Quality Evaluation ({len(questions)} questions)")
        print(f"{'=' * 60}")

        for i, question in enumerate(questions):
            print(f"\n  [{i + 1}/{len(questions)}] {question.id} ({question.category})")
            print(f"    Query: {question.query[:80]}...")

            start = time.perf_counter()
            model_output = ""
            provider_used = ""

            try:
                if use_orchestrator:
                    orch = self._init_orchestrator()
                    task = TaskRequest(
                        source_agent="evaluator",
                        query=question.query,
                    )
                    context = AgentContext(session_id=str(uuid.uuid4()))
                    response = await orch.process(task, context)
                    model_output = response.result or response.error or ""
                    provider_used = "orchestrator"
                else:
                    response = await router.generate(
                        LLMRequest(
                            messages=[LLMMessage(role="user", content=question.query)],
                            system_prompt=(
                                "You are a precise AI assistant specializing in supply chain "
                                "management and drug discovery. Answer accurately and concisely. "
                                "Think step by step before answering."
                            ),
                            temperature=0.5,
                            max_tokens=1024,
                        )
                    )
                    model_output = response.content
                    provider_used = response.provider

                latency_ms = (time.perf_counter() - start) * 1000

                # Metrics from response
                ttft_ms = 0.0
                tps = 0.0
                if hasattr(response, "inference_metrics") and response.inference_metrics:
                    ttft_ms = response.inference_metrics.ttft_ms
                    tps = response.inference_metrics.tokens_per_second

                # Score components
                keyword_score = compute_keyword_score(
                    model_output, question.expected_keywords
                )

                print(f"    ✅ Response ({len(model_output)} chars, {latency_ms:.0f}ms)")
                print(f"    Keywords: {keyword_score:.1f}/10")
                print(f"    Judging...")

                judge_scores = await llm_judge_evaluate(
                    question, model_output, router
                )
                latency_score = compute_latency_score(latency_ms)
                weighted = compute_weighted_score(keyword_score, judge_scores, latency_score)

                print(
                    f"    Scores → Acc:{judge_scores.accuracy:.0f} "
                    f"Rel:{judge_scores.relevance:.0f} "
                    f"Reas:{judge_scores.reasoning:.0f} "
                    f"Halluc:{judge_scores.hallucination:.0f} "
                    f"Lat:{latency_score:.0f} "
                    f"| Weighted: {weighted:.2f}/10"
                )

                results.append(EvalResult(
                    question_id=question.id,
                    category=question.category,
                    query=question.query,
                    model_output=model_output,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    tokens_per_second=tps,
                    keyword_match_score=keyword_score,
                    judge_scores=judge_scores,
                    latency_score=latency_score,
                    weighted_score=weighted,
                    status="pass",
                    provider=provider_used,
                ))

            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                print(f"    ❌ Failed: {e}")
                results.append(EvalResult(
                    question_id=question.id,
                    category=question.category,
                    query=question.query,
                    model_output="",
                    latency_ms=latency_ms,
                    status="fail",
                    error=str(e),
                ))

        return aggregate_results(results)

    # =========================================================================
    # Performance evaluation
    # =========================================================================

    async def run_performance_eval(self) -> dict[str, Any]:
        """Run performance benchmarks."""
        from chainmind.eval.performance import PerformanceEvaluator

        print(f"\n{'=' * 60}")
        print(f"  ChainMind Performance Benchmark")
        print(f"{'=' * 60}")

        model_name = self._settings.local_vllm_served_name
        base_url = self._settings.local_vllm_base_url

        evaluator = PerformanceEvaluator(
            base_url=base_url,
            model_name=model_name,
        )

        report = await evaluator.run_full_benchmark()
        return report.to_dict()

    # =========================================================================
    # RAG evaluation
    # =========================================================================

    async def run_rag_eval(self) -> dict[str, Any]:
        """Run RAG evaluation."""
        from chainmind.eval.rag_eval import RAGEvaluator

        print(f"\n{'=' * 60}")
        print(f"  ChainMind RAG Evaluation")
        print(f"{'=' * 60}")

        router = self._init_llm_router()

        # Try to initialize retriever
        retriever = None
        try:
            from chainmind.retrieval.hybrid_retriever import HybridRetriever
            from chainmind.retrieval.bm25_retriever import BM25Retriever
            from chainmind.retrieval.dense_retriever import DenseRetriever

            bm25 = BM25Retriever()
            dense = DenseRetriever(
                model_name=self._settings.embedding_model,
                persist_dir=self._settings.chromadb_persist_dir,
            )
            retriever = HybridRetriever(retrievers=[bm25, dense])
            print("  ✅ Hybrid retriever initialized")
        except Exception as e:
            print(f"  ⚠️ Retriever initialization failed: {e}")
            print("  Running with LLM-only evaluation (no retrieval metrics)")

        evaluator = RAGEvaluator(retriever=retriever, llm_router=router)
        summary = await evaluator.run_full_evaluation()
        return summary.to_dict()

    # =========================================================================
    # ChainMind-Bench (100-task benchmark)
    # =========================================================================

    async def run_bench(
        self,
        system: str = "chainmind_qwen",
        n_tasks: int | None = None,
        category: str = "all",
    ) -> dict[str, Any]:
        """
        Run ChainMind-Bench via bench_runner.BenchRunner.

        Parameters
        ----------
        system     : one of qwen_direct | chainmind_qwen | react_qwen |
                     gpt4_direct | chainmind_gpt4 | all
        n_tasks    : if set, run only the first N tasks (sample mode)
        category   : filter to A / B / C / D / all
        """
        from chainmind.eval.bench_runner import BenchRunner, SYSTEMS, load_benchmark

        print(f"\n{'=' * 60}")
        print(f"  ChainMind-Bench Evaluation")
        print(f"  System(s): {system}")
        print(f"{'=' * 60}")

        systems = list(SYSTEMS.keys()) if system == "all" else [system]
        tasks = load_benchmark()
        if category != "all":
            tasks = [t for t in tasks if t["category"] == category]
        if n_tasks is not None:
            tasks = tasks[:n_tasks]

        bench = BenchRunner(self._settings)
        raw = await bench.run(systems, tasks)
        tables = bench.compute_tables(raw)
        bench.print_tables(tables)

        # Save bench-specific outputs
        from pathlib import Path
        import datetime
        bench_dir = Path(self._output_dir) / "bench"
        bench_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import json
        raw_path = bench_dir / f"bench_{ts}.json"
        with open(raw_path, "w") as f:
            json.dump({"raw": raw, "tables": tables}, f, indent=2, default=str)
        md = bench.generate_markdown(tables, raw)
        (bench_dir / f"bench_{ts}.md").write_text(md)
        print(f"\n  💾 Bench results → {raw_path}")

        return {"raw": raw, "tables": tables}

    # =========================================================================
    # A/B testing
    # =========================================================================

    async def run_ab_test(self) -> dict[str, Any]:
        """Run prompt A/B testing."""
        from chainmind.eval.prompt_ab import (
            ABTestRunner,
            VARIANT_BASELINE,
            VARIANT_STRUCTURED,
            VARIANT_FEW_SHOT,
        )
        from chainmind.eval.dataset import QUICK_EVAL_QUESTIONS

        print(f"\n{'=' * 60}")
        print(f"  ChainMind Prompt A/B Testing")
        print(f"{'=' * 60}")

        router = self._init_llm_router()
        runner = ABTestRunner(llm_router=router)

        # Use quick eval questions for A/B testing
        questions = QUICK_EVAL_QUESTIONS

        print(f"\n  Test 1: Baseline vs Structured CoT ({len(questions)} questions)")
        result_1 = await runner.run_ab_test(
            VARIANT_BASELINE, VARIANT_STRUCTURED, questions
        )

        print(f"\n  Test 2: Baseline vs Few-Shot ({len(questions)} questions)")
        result_2 = await runner.run_ab_test(
            VARIANT_BASELINE, VARIANT_FEW_SHOT, questions
        )

        return {
            "baseline_vs_structured": result_1.to_dict(),
            "baseline_vs_fewshot": result_2.to_dict(),
        }

    # =========================================================================
    # Report generation
    # =========================================================================

    def _save_report(self, data: dict[str, Any], prefix: str) -> str:
        """Save evaluation report to JSON and markdown."""
        os.makedirs(self._output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON report
        json_path = os.path.join(self._output_dir, f"{prefix}_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        # Markdown report
        md_path = os.path.join(self._output_dir, f"{prefix}_{timestamp}.md")
        with open(md_path, "w") as f:
            f.write(self._generate_markdown_report(data, prefix))

        print(f"\n  📄 Reports saved:")
        print(f"     JSON: {json_path}")
        print(f"     MD:   {md_path}")
        return json_path

    def _generate_markdown_report(self, data: dict[str, Any], title: str) -> str:
        """Generate a markdown evaluation report."""
        lines = [
            f"# ChainMind Evaluation Report: {title}",
            f"",
            f"**Date:** {datetime.datetime.now().isoformat()}",
            f"**Model:** {self._settings.local_vllm_model}",
            f"",
        ]

        if "quality" in data:
            q = data["quality"]
            lines.extend([
                "## Quality Evaluation",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total Questions | {q.get('total_questions', 'N/A')} |",
                f"| Pass Rate | {q.get('pass_rate', 'N/A')} |",
                f"| Avg Weighted Score | {q.get('avg_weighted_score', 'N/A')}/10 |",
                f"| Avg Latency | {q.get('avg_latency_ms', 'N/A')}ms |",
                f"| Avg Tokens/sec | {q.get('avg_tokens_per_second', 'N/A')} |",
                "",
            ])

            if "category_scores" in q:
                lines.extend([
                    "### Scores by Category",
                    "",
                    "| Category | Avg Score |",
                    "|----------|-----------|",
                ])
                for cat, score in q["category_scores"].items():
                    lines.append(f"| {cat} | {score}/10 |")
                lines.append("")

        if "performance" in data:
            p = data["performance"]
            sr = p.get("single_request", {})
            lines.extend([
                "## Performance Benchmark",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| GPU | {p.get('gpu', {}).get('name', 'N/A')} |",
                f"| VRAM Used | {p.get('gpu', {}).get('memory_used_mb', 'N/A')} MiB |",
                f"| Single Request TTFT | {sr.get('ttft_ms', 'N/A')}ms |",
                f"| Single Request E2E | {sr.get('e2e_ms', 'N/A')}ms |",
                f"| Tokens/sec | {sr.get('tokens_per_sec', 'N/A')} |",
                "",
            ])

            if "concurrency_tests" in p:
                lines.extend([
                    "### Concurrency Tests",
                    "",
                    "| Concurrency | Avg TTFT | P95 TTFT | Avg E2E | Throughput |",
                    "|-------------|----------|----------|---------|------------|",
                ])
                for ct in p["concurrency_tests"]:
                    lines.append(
                        f"| {ct['concurrency']} | {ct['avg_ttft_ms']}ms | "
                        f"{ct['p95_ttft_ms']}ms | {ct['avg_e2e_ms']}ms | "
                        f"{ct['total_throughput_tps']} tok/s |"
                    )
                lines.append("")

        if "rag" in data:
            r = data["rag"]
            lines.extend([
                "## RAG Evaluation",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Queries | {r.get('total_queries', 'N/A')} |",
                f"| Avg Recall@K | {r.get('retrieval', {}).get('avg_recall@k', 'N/A')} |",
                f"| Avg Precision@K | {r.get('retrieval', {}).get('avg_precision@k', 'N/A')} |",
                f"| Avg Faithfulness | {r.get('generation', {}).get('avg_faithfulness', 'N/A')}/10 |",
                f"| Avg Attribution | {r.get('generation', {}).get('avg_attribution', 'N/A')}/10 |",
                "",
            ])

        if "ab_tests" in data:
            lines.extend(["## A/B Test Results", ""])
            for test_name, result in data["ab_tests"].items():
                lines.extend([
                    f"### {test_name}",
                    "",
                    f"| Metric | {result.get('variant_a', 'A')} | {result.get('variant_b', 'B')} |",
                    f"|--------|------|------|",
                    f"| Avg Score | {result.get('variant_a_avg_score', 'N/A')} | {result.get('variant_b_avg_score', 'N/A')} |",
                    f"| Avg Latency | {result.get('variant_a_avg_latency_ms', 'N/A')}ms | {result.get('variant_b_avg_latency_ms', 'N/A')}ms |",
                    f"| Wins | {result.get('variant_a_wins', 0)} | {result.get('variant_b_wins', 0)} |",
                    f"| **Winner** | **{result.get('winner', 'N/A')}** | p={result.get('p_value', 'N/A')} |",
                    "",
                ])

        return "\n".join(lines)

    # =========================================================================
    # Main entry point
    # =========================================================================

    async def run(
        self,
        mode: str = "quick",
        bench_system: str = "chainmind_qwen",
        bench_n: int | None = None,
        bench_category: str = "all",
    ) -> dict[str, Any]:
        """Run evaluation in the specified mode."""
        report: dict[str, Any] = {
            "mode": mode,
            "model": self._settings.local_vllm_model,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # ── ChainMind-Bench modes ────────────────────────────────────────────
        if mode == "bench":
            # Sample mode: 10 tasks, one system
            bench = await self.run_bench(
                system=bench_system,
                n_tasks=bench_n if bench_n is not None else 10,
                category=bench_category,
            )
            report["bench"] = bench

        elif mode == "bench-full":
            # Full 100 tasks, all (or chosen) systems
            bench = await self.run_bench(
                system=bench_system,
                n_tasks=bench_n,
                category=bench_category,
            )
            report["bench"] = bench

        # ── Legacy quality modes ─────────────────────────────────────────────
        elif mode in ("full", "quick", "quality"):
            questions = QUICK_EVAL_QUESTIONS if mode == "quick" else ALL_QUESTIONS
            summary = await self.run_quality_eval(questions)
            report["quality"] = summary.to_dict()

        if mode in ("full", "performance"):
            perf = await self.run_performance_eval()
            report["performance"] = perf

        if mode in ("full", "rag"):
            rag = await self.run_rag_eval()
            report["rag"] = rag

        if mode in ("full", "ab"):
            ab = await self.run_ab_test()
            report["ab_tests"] = ab

        # Save report
        self._save_report(report, f"eval_{mode}")

        # Print summary
        self._print_summary(report)

        return report

    def _print_summary(self, report: dict[str, Any]) -> None:
        """Print a concise summary to console."""
        print(f"\n{'=' * 60}")
        print(f"  EVALUATION SUMMARY")
        print(f"{'=' * 60}")

        if "bench" in report:
            tables = report["bench"].get("tables", {})
            t1 = tables.get("table1_tsr", {}).get("rows", {})
            for sys_label, row in t1.items():
                print(f"  Bench TSR [{sys_label}]: {row.get('avg', 'N/A')}%")

        if "quality" in report:
            q = report["quality"]
            print(f"  Quality:  {q.get('avg_weighted_score', 'N/A')}/10 "
                  f"({q.get('passed', 0)}/{q.get('total_questions', 0)} passed)")

        if "performance" in report:
            p = report["performance"]
            sr = p.get("single_request", {})
            print(f"  TTFT:     {sr.get('ttft_ms', 'N/A')}ms")
            print(f"  Speed:    {sr.get('tokens_per_sec', 'N/A')} tok/s")

        if "rag" in report:
            r = report["rag"]
            ret = r.get("retrieval", {})
            gen = r.get("generation", {})
            print(f"  RAG R@K:  {ret.get('avg_recall@k', 'N/A')}")
            print(f"  RAG Faith: {gen.get('avg_faithfulness', 'N/A')}/10")

        if "ab_tests" in report:
            for name, result in report["ab_tests"].items():
                print(f"  A/B [{name}]: Winner = {result.get('winner', 'N/A')}")

        print(f"{'=' * 60}\n")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ChainMind Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full         Run all evaluations (quality + performance + RAG + A/B)
  quick        Quick quality eval (10 questions)
  quality      Quality-only evaluation (55 questions)
  performance  Performance benchmark only
  rag          RAG evaluation only
  ab           Prompt A/B testing only
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["bench", "bench-full", "full", "quick", "quality",
                 "performance", "rag", "ab"],
        default="bench",
        help="Evaluation mode (default: bench)",
    )
    parser.add_argument(
        "--bench-system",
        default="chainmind_qwen",
        help="System for bench mode (default: chainmind_qwen). Use 'all' for all baselines.",
    )
    parser.add_argument(
        "--bench-n",
        type=int,
        default=None,
        help="Max tasks for bench mode (default: 10 for 'bench', all for 'bench-full').",
    )
    parser.add_argument(
        "--bench-category",
        choices=["A", "B", "C", "D", "all"],
        default="all",
        help="Task category filter for bench mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for evaluation reports (default: results)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = EvalRunner(output_dir=args.output_dir)
    asyncio.run(runner.run(
        mode=args.mode,
        bench_system=args.bench_system,
        bench_n=args.bench_n,
        bench_category=args.bench_category,
    ))


if __name__ == "__main__":
    main()
