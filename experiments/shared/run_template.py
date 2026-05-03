"""
Shared run.py template — used by all experiments.
Each experiment's run.py imports this and passes its specific Experiment class.
"""
from __future__ import annotations
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run_experiment(ExperimentClass, default_n: int = 20):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--n", type=int, default=default_n)
    parser.add_argument("--category", choices=["A", "B", "C", "D", "all"], default="all")
    args = parser.parse_args()

    from chainmind.config.settings import Settings
    from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark

    tasks = load_benchmark()
    if args.category != "all":
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.mode == "sample":
        tasks = tasks[: args.n]

    settings = Settings()
    exp = ExperimentClass()
    result = asyncio.run(exp.run(tasks, settings, mode=args.mode))
    result.save()
    return result
