"""
ChainMind Evaluation Dataset — D4 Drug Discovery Questions (D4-only).

This module defines the structured EvalQuestion objects used by the legacy
``runner.py`` quality evaluator.  All 25 supply-chain questions, 10 SC-themed
reasoning questions, and 5 SC-RAG questions have been **removed** — they were
irrelevant to the D4 (Drug Discovery Decision Decomposition) paper scope.

For the full 100-task benchmark (Cat A / B / C / D), use:

    from chainmind.eval.benchmarks.ground_truth_validator import load_benchmark
    tasks = load_benchmark()

Or run the dedicated harness:

    python -m chainmind.eval.bench_runner --mode sample --n 10 --system chainmind_qwen

Contents
--------
D4_QUESTIONS         — 15 structured EvalQuestion objects (chem + web + KG)
ALL_QUESTIONS        — alias for D4_QUESTIONS (preserved for runner.py compat)
QUICK_EVAL_QUESTIONS — 6-question quick-smoke subset
load_bench_questions — loads full 100-task ChainMind-Bench as EvalQuestion list
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


@dataclass
class EvalQuestion:
    """A single evaluation question with metadata."""
    id:               str
    category:         str
    query:            str
    expected_agent:   str
    expected_keywords: list[str]      = field(default_factory=list)
    difficulty:       Difficulty      = Difficulty.MEDIUM
    requires_tools:   bool            = False
    ground_truth:     str | None      = None


# =============================================================================
# Drug Discovery D4 Questions (15)
# =============================================================================

D4_QUESTIONS: list[EvalQuestion] = [

    # ── Computational Chemistry (6) ──────────────────────────────────────────
    EvalQuestion(
        id="D4_CHEM_001",
        category="computational_chemistry",
        query=(
            "Does the molecule with SMILES 'CC(=O)OC1=CC=CC=C1C(=O)O' "
            "(Aspirin) pass Lipinski's Rule of 5?"
        ),
        expected_agent="computational_chemistry",
        expected_keywords=["Lipinski", "MW", "LogP", "HBD", "HBA", "pass"],
        difficulty=Difficulty.EASY,
        requires_tools=True,
        ground_truth="Aspirin passes all Lipinski rules: MW=180.16, LogP≈1.2, HBD=1, HBA=4",
    ),
    EvalQuestion(
        id="D4_CHEM_002",
        category="computational_chemistry",
        query="Find the molecular formula and weight for Imatinib using its name in PubChem.",
        expected_agent="computational_chemistry",
        expected_keywords=["Imatinib", "C29H31N7O", "493", "PubChem"],
        difficulty=Difficulty.EASY,
        requires_tools=True,
        ground_truth="Imatinib: C29H31N7O, MW=493.6 g/mol, CID=5291",
    ),
    EvalQuestion(
        id="D4_CHEM_003",
        category="computational_chemistry",
        query=(
            "Calculate the Tanimoto similarity between Aspirin "
            "(CC(=O)OC1=CC=CC=C1C(=O)O) and Ibuprofen "
            "(CC(C)CC1=CC=C(C=C1)C(C)C(=O)O)."
        ),
        expected_agent="computational_chemistry",
        expected_keywords=["Tanimoto", "similarity", "Morgan", "fingerprint"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_CHEM_004",
        category="computational_chemistry",
        query=(
            "Assess whether Celecoxib "
            "(CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(N)(=O)=O)C(F)(F)F) "
            "is drug-like and list all Lipinski properties."
        ),
        expected_agent="computational_chemistry",
        expected_keywords=["drug-like", "Lipinski", "Celecoxib", "MW"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_CHEM_005",
        category="computational_chemistry",
        query=(
            "What are the key structural differences between selective "
            "COX-2 inhibitors and non-selective NSAIDs at the molecular level?"
        ),
        expected_agent="computational_chemistry",
        expected_keywords=["COX-2", "selectivity", "sulfonamide", "NSAID"],
        difficulty=Difficulty.HARD,
        requires_tools=False,
    ),
    EvalQuestion(
        id="D4_CHEM_006",
        category="computational_chemistry",
        query=(
            "Generate a 3D conformer for Caffeine "
            "(CN1C=NC2=C1C(=O)N(C(=O)N2C)C) and validate the geometry."
        ),
        expected_agent="computational_chemistry",
        expected_keywords=["conformer", "3D", "MMFF94", "caffeine"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),

    # ── Web Research (5) ─────────────────────────────────────────────────────
    EvalQuestion(
        id="D4_WEB_001",
        category="web_research",
        query="What are the latest approaches in using GNNs for molecular property prediction?",
        expected_agent="web_research",
        expected_keywords=["GNN", "molecular", "property", "prediction", "graph"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_WEB_002",
        category="web_research",
        query=(
            "Find recent papers on transformer-based models for ADMET prediction "
            "published in 2024-2025."
        ),
        expected_agent="web_research",
        expected_keywords=["transformer", "ADMET", "paper", "2024"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_WEB_003",
        category="web_research",
        query=(
            "What is the current state-of-the-art model for the "
            "TDC CYP2D6 substrate prediction benchmark?"
        ),
        expected_agent="web_research",
        expected_keywords=["TDC", "CYP2D6", "benchmark", "SOTA"],
        difficulty=Difficulty.HARD,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_WEB_004",
        category="web_research",
        query="Summarize the key findings from AlphaFold3 for drug-target interaction prediction.",
        expected_agent="web_research",
        expected_keywords=["AlphaFold", "drug-target", "interaction", "protein"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_WEB_005",
        category="web_research",
        query=(
            "What are the main challenges in applying reinforcement learning "
            "to de novo drug design?"
        ),
        expected_agent="web_research",
        expected_keywords=["reinforcement learning", "de novo", "drug design", "challenge"],
        difficulty=Difficulty.HARD,
        requires_tools=True,
    ),

    # ── Knowledge Graph (4) ──────────────────────────────────────────────────
    EvalQuestion(
        id="D4_KG_001",
        category="knowledge_graph",
        query=(
            "Explain how Targeted Protein Degradation works using a Knowledge Graph "
            "where PROTAC binds to Target Protein and E3 Ligase."
        ),
        expected_agent="knowledge_graph",
        expected_keywords=["PROTAC", "E3 ligase", "ubiquitin", "degradation"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_KG_002",
        category="knowledge_graph",
        query=(
            "Create a knowledge graph showing the ADMET prediction pipeline: "
            "from SMILES input through featurization to model prediction."
        ),
        expected_agent="knowledge_graph",
        expected_keywords=["ADMET", "SMILES", "featurization", "pipeline"],
        difficulty=Difficulty.MEDIUM,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_KG_003",
        category="knowledge_graph",
        query="Generate a knowledge graph explaining how GNNs process molecular graphs for property prediction.",
        expected_agent="knowledge_graph",
        expected_keywords=["GNN", "molecular graph", "message passing", "node"],
        difficulty=Difficulty.HARD,
        requires_tools=True,
    ),
    EvalQuestion(
        id="D4_KG_004",
        category="knowledge_graph",
        query=(
            "Visualize the relationship between drug resistance mechanisms: "
            "target mutation, efflux pumps, and metabolic inactivation."
        ),
        expected_agent="knowledge_graph",
        expected_keywords=["resistance", "mutation", "efflux", "mechanism"],
        difficulty=Difficulty.HARD,
        requires_tools=True,
    ),
]


# =============================================================================
# Aliases  (backward-compat with runner.py)
# =============================================================================

ALL_QUESTIONS: list[EvalQuestion] = D4_QUESTIONS

# Quick smoke subset — one from each agent type, easy to hard spread
QUICK_EVAL_QUESTIONS: list[EvalQuestion] = [
    q for q in ALL_QUESTIONS
    if q.id in {
        "D4_CHEM_001",  # easy, ground truth, tool required
        "D4_CHEM_003",  # medium, similarity calculation
        "D4_CHEM_006",  # medium, conformer generation
        "D4_WEB_001",   # medium, arxiv search
        "D4_KG_001",    # medium, KG generation
        "D4_WEB_005",   # hard, no-tool reasoning
    }
]


# =============================================================================
# Bridge to ChainMind-Bench (100 tasks)
# =============================================================================

def load_bench_questions() -> list[EvalQuestion]:
    """
    Load the full 100-task ChainMind-Bench as EvalQuestion objects.

    Categories map to expected agents:
      A  → computational_chemistry
      B  → web_research
      C  → knowledge_graph
      D  → orchestrator  (multi-step)

    This allows the legacy runner.py quality evaluator to run on the full
    benchmark with ``--mode bench``.
    """
    import json

    bench_path = (
        Path(__file__).parent / "benchmarks" / "chainmind_bench.json"
    )
    with open(bench_path) as f:
        data: dict[str, Any] = json.load(f)

    _AGENT_MAP = {
        "A": "computational_chemistry",
        "B": "web_research",
        "C": "knowledge_graph",
        "D": "orchestrator",
    }

    questions: list[EvalQuestion] = []
    for task in data.get("tasks", []):
        cat = task.get("category", "A")
        gt_obj = task.get("ground_truth", {})

        # Derive expected_keywords from ground_truth or required_concepts
        keywords: list[str] = []
        if "required_concepts" in gt_obj:
            keywords = gt_obj["required_concepts"]
        elif "expected_keywords" in gt_obj:
            keywords = gt_obj["expected_keywords"]

        questions.append(EvalQuestion(
            id=task["id"],
            category=task.get("subcategory", cat),
            query=task["query"],
            expected_agent=_AGENT_MAP.get(cat, "orchestrator"),
            expected_keywords=keywords,
            difficulty=Difficulty(task.get("difficulty", "medium")),
            requires_tools=bool(task.get("requires_tool") or task.get("requires_tools")),
            ground_truth=json.dumps(gt_obj) if gt_obj else None,
        ))

    return questions


# =============================================================================
# Utility filters
# =============================================================================

def get_questions_by_category(category: str) -> list[EvalQuestion]:
    """Filter D4 questions by category string."""
    return [q for q in ALL_QUESTIONS if q.category == category]


def get_questions_by_difficulty(difficulty: Difficulty) -> list[EvalQuestion]:
    """Filter D4 questions by difficulty level."""
    return [q for q in ALL_QUESTIONS if q.difficulty == difficulty]
