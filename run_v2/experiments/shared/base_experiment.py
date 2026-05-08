"""
experiments/shared/base_experiment.py
Shared base class for all ChainMind experiments.

Every experiment:
1. Extends BaseExperiment
2. Overrides build_agent() to return a modified agent
3. Can optionally override pre/post hooks
4. Outputs a standardised ExperimentResult

Run any experiment standalone:
    python experiments/exp004_few_shot/run.py --n 20
    python experiments/exp004_few_shot/run.py --mode full
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
RESULTS_ROOT = Path("results/experiments")


# ---------------------------------------------------------------------------
# Result schema (shared across all experiments)
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_id: str
    category: str
    subcategory: str
    difficulty: str
    experiment: str
    score: float
    passed: bool
    latency_ms: float
    error_type: str
    response_preview: str = ""
    breakdown: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str          # e.g. "exp004_few_shot"
    paper_ref: str              # BibTeX-style cite key
    hypothesis: str
    timestamp: str
    mode: str                   # "sample" | "full"
    n_tasks: int
    task_results: list[TaskResult]

    # Aggregate metrics (computed post-run)
    tsr_overall: float = 0.0
    tsr_by_category: dict = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    n_passed: int = 0

    def compute_aggregates(self) -> None:
        n = max(len(self.task_results), 1)
        self.n_passed = sum(1 for t in self.task_results if t.passed)
        self.tsr_overall = round(self.n_passed / n * 100, 2)
        self.avg_latency_ms = round(
            sum(t.latency_ms for t in self.task_results) / n, 1
        )
        cats = {}
        for t in self.task_results:
            cats.setdefault(t.category, []).append(t.passed)
        self.tsr_by_category = {
            cat: round(sum(v) / len(v) * 100, 1)
            for cat, v in cats.items()
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  {self.experiment_id}")
        print(f"  Paper: {self.paper_ref}")
        print(f"{'='*60}")
        print(f"  TSR Overall : {self.tsr_overall:.1f}%  ({self.n_passed}/{self.n_tasks} passed)")
        print(f"  By category : {self.tsr_by_category}")
        print(f"  Avg latency : {self.avg_latency_ms:.0f} ms")
        errs = {}
        for t in self.task_results:
            errs[t.error_type] = errs.get(t.error_type, 0) + 1
        print(f"  Error types : {errs}")

    def save(self, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or (RESULTS_ROOT / self.experiment_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"result_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"  💾 Saved → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# Base experiment class
# ---------------------------------------------------------------------------

class BaseExperiment(ABC):
    """
    Abstract base for all ChainMind experiments.

    Subclasses implement:
        experiment_id   — unique ID string
        paper_ref       — citation key for the paper being replicated
        hypothesis      — one-sentence hypothesis
        build_agent()   — return a configured (possibly modified) agent
    """

    @property
    @abstractmethod
    def experiment_id(self) -> str: ...

    @property
    @abstractmethod
    def paper_ref(self) -> str: ...

    @property
    @abstractmethod
    def hypothesis(self) -> str: ...

    @abstractmethod
    def build_orchestrator(self, settings: Any):
        """Return a configured orchestrator (or agent) for this experiment."""
        ...

    async def run_task(self, orchestrator: Any, task: dict) -> TaskResult:
        """Run a single task through the experiment's orchestrator."""
        from chainmind.core.types import AgentContext, TaskRequest
        from chainmind.eval.benchmarks.ground_truth_validator import score_response

        start = time.perf_counter()
        try:
            task_req = TaskRequest(
                source_agent=self.experiment_id,
                query=task["query"],
            )
            ctx = AgentContext(session_id=str(uuid.uuid4()))
            resp = await orchestrator.process(task_req, ctx)
            response_text = resp.result or resp.error or ""
        except Exception as e:
            response_text = f"ERROR: {e}"
            logger.warning(f"[{self.experiment_id}] Task {task['id']} error: {e}")

        latency_ms = (time.perf_counter() - start) * 1000
        score = score_response(task, response_text)

        from chainmind.eval.bench_runner import classify_error
        error_type = classify_error(response_text, task, score["score"])

        return TaskResult(
            task_id=task["id"],
            category=task["category"],
            subcategory=task.get("subcategory", ""),
            difficulty=task.get("difficulty", "medium"),
            experiment=self.experiment_id,
            score=score["score"],
            passed=score["passed"],
            latency_ms=round(latency_ms, 1),
            error_type=error_type,
            response_preview=response_text[:200],
            breakdown=score.get("breakdown", {}),
        )

    async def run(
        self,
        tasks: list[dict],
        settings: Any,
        mode: str = "sample",
    ) -> ExperimentResult:
        """Run the full experiment on the given task list."""
        print(f"\n{'='*60}")
        print(f"  EXP: {self.experiment_id}")
        print(f"  H:   {self.hypothesis[:80]}...")
        print(f"  N:   {len(tasks)} tasks")
        print(f"{'='*60}")

        orchestrator = self.build_orchestrator(settings)
        task_results = []

        for i, task in enumerate(tasks):
            print(f"  [{i+1:3d}/{len(tasks)}] {task['id']:6s} | ", end="", flush=True)
            result = await self.run_task(orchestrator, task)
            icon = "✅" if result.passed else "❌"
            print(f"{icon} {result.score:.2f} | {result.latency_ms:.0f}ms | {result.error_type}")
            task_results.append(result)

        exp_result = ExperimentResult(
            experiment_id=self.experiment_id,
            paper_ref=self.paper_ref,
            hypothesis=self.hypothesis,
            timestamp=datetime.datetime.now().isoformat(),
            mode=mode,
            n_tasks=len(tasks),
            task_results=task_results,
        )
        exp_result.compute_aggregates()
        exp_result.print_summary()
        return exp_result
