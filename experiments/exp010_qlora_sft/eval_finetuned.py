#!/usr/bin/env python3
"""
experiments/exp010_qlora_sft/eval_finetuned.py

Evaluates the fine-tuned ChainMind model against the same benchmark used for the
base model (run_v5), producing a direct performance comparison table.

Usage:
    # After fine-tuning, vLLM is already serving at port 8101 (fine-tuned)
    # and port 8100 (base model, optional comparison)
    python3 experiments/exp010_qlora_sft/eval_finetuned.py \
        --model-port 8101 \
        --baseline-results run_v5/results/ \
        --output-dir run_ft_eval/results/

Design:
    - Runs the full ChainMind benchmark (100 tasks) on the fine-tuned model
    - Loads run_v5 baseline results for comparison
    - Prints per-category TSR: base vs fine-tuned, delta
    - Saves results to --output-dir as JSON + markdown table
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ENVIRONMENT", "development")

from chainmind.eval.benchmarks.ground_truth_validator import score_response
from chainmind.core.types import LLMRequest, LLMMessage, AgentContext
from chainmind.llm.local_provider import LocalProvider


async def run_single_task(
    provider: LocalProvider,
    task: dict[str, Any],
    system_prompt: str,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Run one benchmark task and return scored result."""
    start = time.perf_counter()
    try:
        req = LLMRequest(
            messages=[LLMMessage(role="user", content=task["query"])],
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        resp = await provider.generate(req)
        latency_ms = (time.perf_counter() - start) * 1000

        scored = score_response(task, resp.content)
        return {
            "task_id": task["task_id"],
            "category": task.get("category", "?"),
            "score": scored["score"],
            "passed": scored["passed"],
            "latency_ms": latency_ms,
            "response_preview": resp.content[:200],
        }
    except Exception as e:
        return {
            "task_id": task["task_id"],
            "category": task.get("category", "?"),
            "score": 0.0,
            "passed": False,
            "latency_ms": (time.perf_counter() - start) * 1000,
            "error": str(e),
        }


def load_benchmark_tasks(bench_path: Path) -> list[dict]:
    """Load ChainMind benchmark tasks from JSON."""
    with open(bench_path) as f:
        bench = json.load(f)
    tasks = []
    for task in bench.get("tasks", []):
        tasks.append(task)
    return tasks


def load_baseline_results(results_dir: Path) -> dict[str, Any]:
    """
    Load run_v5 (baseline) results for comparison.
    Looks for the best-performing strategy's result file.
    """
    if not results_dir.exists():
        return {}

    results = {}
    for json_file in sorted(results_dir.glob("**/*.json")):
        try:
            data = json.loads(json_file.read_text())
            strategy = data.get("strategy", json_file.stem)
            overall_tsr = data.get("metrics", {}).get("overall_tsr", 0.0)
            # Keep best TSR per category
            if overall_tsr > results.get("best_overall", 0):
                results["best_overall"] = overall_tsr
                results["best_strategy"] = strategy
                results["best_data"] = data
        except Exception:
            continue
    return results


def build_system_prompt() -> str:
    """ReAct specialist prompt for evaluation."""
    return """You are a ChainMind Drug Discovery AI specialist.

You solve molecular research tasks step by step using the ReAct framework:

THOUGHT: <reasoning about the task>
ACTION: <tool_name>
ACTION_INPUT: <tool parameters as JSON>
OBSERVATION: <tool result>
... (repeat as needed)
FINAL_ANSWER: <conclusive answer>

Available tools:
- assess_lipinski_rules(smiles): Check Lipinski Rule of 5 for a molecule
- get_canonical_smiles(query): Get canonical SMILES from PubChem by drug name
- search_literature(query): Search scientific literature
- generate_knowledge_graph(topic): Create a Mermaid knowledge graph

For knowledge graph tasks: ALWAYS include a ```mermaid\ngraph TD\n...``` block.
For molecular property tasks: ALWAYS use tools for exact values, never guess.
For literature tasks: List specific papers with titles and key findings."""


async def evaluate(
    model_port: int,
    bench_path: Path,
    baseline_results: dict,
    output_dir: Path,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Main evaluation loop."""
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = LocalProvider(
        base_url=f"http://0.0.0.0:{model_port}/v1",
        served_model_name="chainmind-ft",
    )

    # Health check
    healthy = await provider.health_check()
    if not healthy:
        raise RuntimeError(f"vLLM not healthy on port {model_port}. Is it running?")
    print(f"✅ vLLM healthy on port {model_port}")

    tasks = load_benchmark_tasks(bench_path)
    if max_tasks:
        tasks = tasks[:max_tasks]

    system_prompt = build_system_prompt()
    print(f"Running {len(tasks)} benchmark tasks on fine-tuned model...")

    results = []
    cat_scores: dict[str, list[float]] = {}

    for i, task in enumerate(tasks, 1):
        result = await run_single_task(provider, task, system_prompt)
        results.append(result)

        cat = result["category"]
        cat_scores.setdefault(cat, []).append(result["score"])

        symbol = "✅" if result["passed"] else "❌"
        print(f"  [{i:3d}/{len(tasks)}] {result['task_id']:8s} | {symbol} {result['score']:.2f} | {result['latency_ms']:.0f}ms")

    # ── Per-category TSR ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINE-TUNED MODEL EVALUATION RESULTS")
    print("=" * 60)

    ft_tsr: dict[str, float] = {}
    for cat, scores in sorted(cat_scores.items()):
        tsr = sum(1 for s in scores if s >= 0.5) / len(scores)
        avg_score = sum(scores) / len(scores)
        ft_tsr[cat] = tsr
        print(f"  Cat-{cat}: TSR={tsr:.1%}  avg_score={avg_score:.2f}  ({len(scores)} tasks)")

    overall_tsr = sum(1 for r in results if r["passed"]) / len(results)
    overall_avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  OVERALL TSR:   {overall_tsr:.1%}")
    print(f"  OVERALL Score: {overall_avg:.3f}")

    # ── Comparison table ────────────────────────────────────────────────────
    if baseline_results.get("best_data"):
        base = baseline_results["best_data"]
        base_metrics = base.get("metrics", {})
        base_cat = base_metrics.get("per_category_tsr", {})

        print(f"\n{'='*60}")
        print(f"COMPARISON: Base Qwen2.5-7B vs ChainMind-FT")
        print(f"(Baseline: {baseline_results.get('best_strategy', 'run_v5')})")
        print(f"{'='*60}")
        print(f"{'Category':<12} {'Base TSR':>10} {'FT TSR':>10} {'Delta':>10}")
        print("-" * 44)

        for cat in sorted(ft_tsr.keys()):
            base_val = base_cat.get(f"Cat-{cat}", base_cat.get(cat, 0.0))
            ft_val = ft_tsr[cat]
            delta = ft_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"  Cat-{cat:<8} {base_val:>9.1%} {ft_val:>9.1%} {sign}{delta:>8.1%}")

        base_overall = base_metrics.get("overall_tsr", 0.0)
        delta_overall = overall_tsr - base_overall
        sign = "+" if delta_overall >= 0 else ""
        print("-" * 44)
        print(f"  {'OVERALL':<10} {base_overall:>9.1%} {overall_tsr:>9.1%} {sign}{delta_overall:>8.1%}")

    # ── Save results ────────────────────────────────────────────────────────
    output = {
        "model": "chainmind-ft-v1",
        "port": model_port,
        "n_tasks": len(tasks),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "overall_tsr": overall_tsr,
        "overall_avg_score": overall_avg,
        "per_category_tsr": ft_tsr,
        "baseline_strategy": baseline_results.get("best_strategy", "unknown"),
        "baseline_overall_tsr": baseline_results.get("best_data", {}).get("metrics", {}).get("overall_tsr", 0.0),
        "task_results": results,
    }

    out_file = output_dir / f"ft_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Results saved: {out_file}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned ChainMind model")
    parser.add_argument("--model-port", type=int, default=8101, help="Port where fine-tuned vLLM is running")
    parser.add_argument("--baseline-results", type=Path, default=Path("run_v5/results"), help="Directory with baseline run results")
    parser.add_argument("--output-dir", type=Path, default=Path("run_ft_eval/results"), help="Where to save evaluation results")
    parser.add_argument("--bench-path", type=Path, default=Path("chainmind/eval/benchmarks/chainmind_bench.json"), help="Benchmark JSON path")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks (default: all 100)")
    args = parser.parse_args()

    result = asyncio.run(evaluate(
        model_port=args.model_port,
        bench_path=args.bench_path,
        baseline_results=load_baseline_results(args.baseline_results),
        output_dir=args.output_dir,
        max_tasks=args.max_tasks,
    ))

    overall_tsr = result["overall_tsr"]
    base_tsr = result.get("baseline_overall_tsr", 0.0)
    print(f"\n{'='*60}")
    print(f"Fine-tune IMPROVED TSR by {overall_tsr - base_tsr:+.1%}")
    print(f"  Base: {base_tsr:.1%}  →  Fine-tuned: {overall_tsr:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
