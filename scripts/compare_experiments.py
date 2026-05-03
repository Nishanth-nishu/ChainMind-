#!/usr/bin/env python3
"""
scripts/compare_experiments.py
Compare all experiment results and produce a ranked TSR table.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --results-dir results/experiments
    python scripts/compare_experiments.py --format markdown
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_latest_result(exp_dir: Path) -> dict | None:
    """Load the most recent result JSON from an experiment directory."""
    jsons = sorted(exp_dir.glob("result_*.json"), reverse=True)
    if not jsons:
        return None
    with open(jsons[0]) as f:
        return json.load(f)


def gather_results(results_root: Path) -> list[dict]:
    """Gather latest result from each experiment directory."""
    rows = []
    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        data = load_latest_result(exp_dir)
        if data is None:
            continue
        by_cat = data.get("tsr_by_category", {})
        rows.append({
            "exp": data.get("experiment_id", exp_dir.name),
            "paper": data.get("paper_ref", "—"),
            "n": data.get("n_tasks", 0),
            "tsr": data.get("tsr_overall", 0.0),
            "cat_A": by_cat.get("A", "—"),
            "cat_B": by_cat.get("B", "—"),
            "cat_C": by_cat.get("C", "—"),
            "cat_D": by_cat.get("D", "—"),
            "latency_ms": data.get("avg_latency_ms", 0),
            "timestamp": data.get("timestamp", "")[:19],
        })

    # Sort by TSR descending
    rows.sort(key=lambda r: r["tsr"], reverse=True)
    return rows


def print_table(rows: list[dict], fmt: str = "text") -> None:
    """Print comparison table (text or markdown)."""
    if not rows:
        print("No results found. Run experiments first:\n  bash scripts/run_all_experiments.sh")
        return

    if fmt == "markdown":
        print("\n## ChainMind Experiment Comparison\n")
        print("| Rank | Experiment | TSR% | Cat-A | Cat-B | Cat-C | Cat-D | Avg Latency | N | Paper |")
        print("|------|-----------|------|-------|-------|-------|-------|-------------|---|-------|")
        for i, r in enumerate(rows, 1):
            print(
                f"| {i} | `{r['exp']}` | **{r['tsr']:.1f}%** "
                f"| {r['cat_A']} | {r['cat_B']} | {r['cat_C']} | {r['cat_D']} "
                f"| {r['latency_ms']:.0f}ms | {r['n']} | {r['paper']} |"
            )
    else:
        header = f"{'Rank':<5} {'Experiment':<35} {'TSR%':>6} {'A%':>6} {'B%':>6} {'C%':>6} {'D%':>6} {'Lat(ms)':>9} {'N':>4}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        for i, r in enumerate(rows, 1):
            a = f"{r['cat_A']:.1f}" if isinstance(r['cat_A'], float) else str(r['cat_A'])
            b = f"{r['cat_B']:.1f}" if isinstance(r['cat_B'], float) else str(r['cat_B'])
            c = f"{r['cat_C']:.1f}" if isinstance(r['cat_C'], float) else str(r['cat_C'])
            d = f"{r['cat_D']:.1f}" if isinstance(r['cat_D'], float) else str(r['cat_D'])
            print(
                f"  {i:<3} {r['exp']:<35} {r['tsr']:>5.1f}% "
                f"{a:>6} {b:>6} {c:>6} {d:>6} "
                f"{r['latency_ms']:>8.0f} {r['n']:>4}"
            )
        print("=" * len(header))

    # Best experiment summary
    if rows:
        best = rows[0]
        print(f"\n🏆  Best: {best['exp']}  TSR={best['tsr']:.1f}%  ({best['paper']})")
        if len(rows) > 1:
            baseline = rows[-1]
            gain = best["tsr"] - baseline["tsr"]
            print(f"📈  Gain over worst: +{gain:.1f}pp")

    # Show error taxonomy from best experiment
    print("\n--- Error taxonomy of best experiment ---")
    if rows:
        best_dir = Path("results/experiments") / rows[0]["exp"]
        data = load_latest_result(best_dir)
        if data:
            errs: dict[str, int] = {}
            for t in data.get("task_results", []):
                et = t.get("error_type", "None")
                errs[et] = errs.get(et, 0) + 1
            for etype, cnt in sorted(errs.items(), key=lambda x: -x[1]):
                bar = "█" * min(cnt, 30)
                print(f"  {etype:<20} {cnt:>3}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Compare ChainMind experiments")
    parser.add_argument("--results-dir", default="results/experiments")
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"Results directory not found: {results_root}")
        print("Run:  bash scripts/run_all_experiments.sh")
        return

    rows = gather_results(results_root)
    print_table(rows, fmt=args.format)


if __name__ == "__main__":
    main()
