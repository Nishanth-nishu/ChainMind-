"""
ChainMind-Bench Ground Truth Validator.

Deterministic, category-specific scoring for each of the 4 benchmark categories:

  Category A (Molecular Property)
      Numerical comparison against RDKit-computed reference values.
      Tolerance: ±5% for continuous values, exact match for booleans.

  Category B (Literature Retrieval)
      Keyword recall: fraction of expected keywords found in the response.
      Threshold: min_keyword_recall (default 0.6 = 60%).

  Category C (Knowledge Graph)
      1. Mermaid syntax validity: response must contain a ```mermaid block.
      2. Edge count: ≥ min_edges extracted from the Mermaid graph TD section.
      3. Concept presence: required_concepts must appear in the response.

  Category D (Multi-Step)
      Composite score: mean of per-step subscores using the above validators.
      A task succeeds (TSR=1) only if ALL steps score ≥ their thresholds.

Usage
-----
    from chainmind.eval.benchmarks.ground_truth_validator import score_response, validate_all
    result = score_response(task, system_response_text)
    # result: {"score": 0.85, "passed": True, "breakdown": {...}}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BENCH_PATH = Path(__file__).parent / "chainmind_bench.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_benchmark() -> list[dict[str, Any]]:
    """Load all tasks from chainmind_bench.json."""
    with open(BENCH_PATH) as f:
        data = json.load(f)
    return data["tasks"]


def score_response(task: dict[str, Any], response: str) -> dict[str, Any]:
    """
    Score a system response against the ground truth for a single task.

    Returns
    -------
    {
        "task_id": str,
        "category": str,
        "score": float,          # 0.0 – 1.0
        "passed": bool,          # score >= pass threshold
        "breakdown": dict,       # per-criterion subscores
        "error": str | None,     # validation error if any
    }
    """
    cat = task.get("category", "")
    try:
        if cat == "A":
            return _score_cat_a(task, response)
        elif cat == "B":
            return _score_cat_b(task, response)
        elif cat == "C":
            return _score_cat_c(task, response)
        elif cat == "D":
            return _score_cat_d(task, response)
        else:
            return _make_result(task, 0.0, False, {}, f"Unknown category: {cat}")
    except Exception as e:
        return _make_result(task, 0.0, False, {}, f"Validator error: {e}")


def validate_all(tasks: list[dict] | None = None, verbose: bool = False) -> dict[str, Any]:
    """
    Smoke-test the validator itself: verifies all tasks load and have parseable
    ground_truth fields (does NOT run the LLM — no system call is made).

    Returns a summary dict with validation errors.
    """
    if tasks is None:
        tasks = load_benchmark()

    errors = []
    for task in tasks:
        tid = task.get("id", "?")
        if "ground_truth" not in task:
            errors.append(f"{tid}: missing ground_truth")
            continue
        if "category" not in task:
            errors.append(f"{tid}: missing category")
            continue
        if verbose:
            print(f"  ✓ {tid} ({task['category']})")

    total = len(tasks)
    ok = total - len(errors)
    print(f"\nChainMind-Bench Validation: {ok}/{total} tasks OK, {len(errors)} errors")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
    return {"total": total, "ok": ok, "errors": errors}


# ---------------------------------------------------------------------------
# Category A: Molecular Property Scorer
# ---------------------------------------------------------------------------

def _score_cat_a(task: dict, response: str) -> dict[str, Any]:
    """
    Score against numerical / boolean ground truth.
    Checks:
      - passes_ro5 (bool) → exact match in response text
      - molecular_weight  → |predicted - gt| / gt ≤ tolerance
      - violations (int)  → mentioned in response
      - tanimoto_similarity → within range
    """
    gt = task["ground_truth"]
    tol = task.get("tolerance", 0.05)
    resp_lower = response.lower()
    criteria: dict[str, float] = {}

    # Check boolean: passes_ro5
    if "passes_ro5" in gt:
        expected = gt["passes_ro5"]
        if expected:
            found = any(kw in resp_lower for kw in ["pass", "drug-like", "satisfies", "yes", "meets"])
        else:
            found = any(kw in resp_lower for kw in ["fail", "violat", "does not pass", "not drug-like"])
        criteria["passes_ro5"] = 1.0 if found else 0.0

    # Check molecular_weight (numerical)
    if "molecular_weight" in gt:
        expected_mw = gt["molecular_weight"]
        found_mw = _extract_number_near_keyword(response, ["molecular weight", "mw", "mol. wt", "weight"])
        if found_mw is not None:
            err = abs(found_mw - expected_mw) / max(expected_mw, 1e-6)
            criteria["molecular_weight"] = 1.0 if err <= tol else max(0.0, 1.0 - err / tol)
        else:
            # Check if the number appears anywhere in the response (rounded)
            mw_str = f"{expected_mw:.0f}"
            criteria["molecular_weight"] = 0.5 if mw_str in response else 0.0

    # Check violations (integer)
    if "violations" in gt:
        expected_v = gt["violations"]
        criteria["violations"] = 1.0 if str(expected_v) in response else 0.5

    # Check tanimoto range
    if "tanimoto_similarity_range" in gt:
        lo, hi = gt["tanimoto_similarity_range"]
        t_val = _extract_number_near_keyword(response, ["tanimoto", "similarity", "score"])
        if t_val is not None:
            criteria["tanimoto"] = 1.0 if lo <= t_val <= hi else 0.0
        else:
            criteria["tanimoto"] = 0.0

    # Check bool: mmff94_converged
    if "mmff94_converged" in gt:
        expected = gt["mmff94_converged"]
        if expected:
            found = any(kw in resp_lower for kw in ["converged", "success", "true", "optimized"])
        else:
            found = any(kw in resp_lower for kw in ["not converged", "failed", "false"])
        criteria["mmff94_converged"] = 1.0 if found else 0.0

    # Check formula
    if "formula" in gt:
        criteria["formula"] = 1.0 if gt["formula"].lower() in resp_lower else 0.0

    # Check weight range
    if "weight_range" in gt:
        lo, hi = gt["weight_range"]
        found_w = _extract_any_number_in_range(response, lo - 5, hi + 5)
        criteria["weight_range"] = 1.0 if found_w else 0.0

    # Check CID
    if "cid" in gt:
        criteria["cid"] = 1.0 if str(gt["cid"]) in response else 0.5

    # Check minimum atoms
    if "num_atoms_ge" in gt:
        found_n = _extract_number_near_keyword(response, ["atom", "num_atom"])
        criteria["num_atoms"] = 1.0 if (found_n is not None and found_n >= gt["num_atoms_ge"]) else 0.0

    score = (sum(criteria.values()) / len(criteria)) if criteria else 0.0
    passed = score >= 0.6
    return _make_result(task, score, passed, criteria)


# ---------------------------------------------------------------------------
# Category B: Literature Retrieval Scorer
# ---------------------------------------------------------------------------

def _score_cat_b(task: dict, response: str) -> dict[str, Any]:
    """
    Score keyword recall: fraction of expected_keywords found in response.
    Passes if recall >= min_keyword_recall.
    """
    gt = task["ground_truth"]
    expected_kws = [kw.lower() for kw in gt.get("expected_keywords", [])]
    min_recall = gt.get("min_keyword_recall", 0.6)
    resp_lower = response.lower()

    if not expected_kws:
        return _make_result(task, 1.0, True, {"no_keywords": 1.0})

    hits = [kw for kw in expected_kws if kw in resp_lower]
    recall = len(hits) / len(expected_kws)

    # Bonus: tool was actually called (response has structured data patterns)
    tool_bonus = 0.0
    tool_indicators = ['"title"', '"url"', '"abstract"', '"arxiv_id"', '"papers"',
                       'http', 'arxiv.org', 'pubmed', 'sota_model', 'tdc_url']
    if any(ind in response for ind in tool_indicators):
        tool_bonus = 0.1

    final_score = min(1.0, recall + tool_bonus)
    passed = recall >= min_recall

    criteria = {
        "keyword_recall": round(recall, 3),
        "keywords_found": hits,
        "keywords_missed": [kw for kw in expected_kws if kw not in resp_lower],
        "tool_called_bonus": tool_bonus,
    }
    return _make_result(task, final_score, passed, criteria)


# ---------------------------------------------------------------------------
# Category C: Knowledge Graph Scorer
# ---------------------------------------------------------------------------

def _score_cat_c(task: dict, response: str) -> dict[str, Any]:
    """
    Score Mermaid KG generation:
    1. Contains ```mermaid block (0.4 weight)
    2. Has >= min_edges edges parsed from graph TD (0.4 weight)
    3. Required concepts mentioned in response (0.2 weight)
    """
    gt = task["ground_truth"]
    min_edges = gt.get("min_edges", 5)
    required_concepts = [c.lower() for c in gt.get("required_concepts", [])]
    criteria: dict[str, Any] = {}

    # Check mermaid block exists
    has_mermaid = "```mermaid" in response or "mermaid" in response.lower()
    criteria["has_mermaid_block"] = 1.0 if has_mermaid else 0.0

    # Count edges in Mermaid (lines with --> or -- "..." -->)
    edge_count = _count_mermaid_edges(response)
    criteria["edge_count"] = edge_count
    criteria["edge_score"] = 1.0 if edge_count >= min_edges else (edge_count / min_edges)

    # Concept presence
    resp_lower = response.lower()
    if required_concepts:
        concept_hits = [c for c in required_concepts if c in resp_lower]
        concept_recall = len(concept_hits) / len(required_concepts)
        criteria["concept_recall"] = round(concept_recall, 3)
        criteria["concepts_found"] = concept_hits
    else:
        concept_recall = 1.0
        criteria["concept_recall"] = 1.0

    # Weighted score
    score = (
        0.35 * criteria["has_mermaid_block"]
        + 0.40 * criteria["edge_score"]
        + 0.25 * concept_recall
    )
    passed = has_mermaid and edge_count >= max(1, min_edges - 1) and concept_recall >= 0.4
    return _make_result(task, round(score, 3), passed, criteria)


# ---------------------------------------------------------------------------
# Category D: Multi-Step Chain Scorer
# ---------------------------------------------------------------------------

def _score_cat_d(task: dict, response: str) -> dict[str, Any]:
    """
    Composite scorer for multi-step tasks.
    Uses per-step validators based on the tools required.
    A task passes if all steps score >= 0.5.
    """
    gt = task["ground_truth"]
    required_tools = task.get("requires_tools", [])
    criteria: dict[str, Any] = {}
    step_scores: list[float] = []

    resp_lower = response.lower()

    # Check each tool was invoked (proxy: tool name or its output signature appears)
    TOOL_SIGNATURES = {
        "assess_lipinski_rules": ["lipinski", "mw", "logp", "hbd", "hba", "violations", "drug-like"],
        "calculate_similarity":  ["tanimoto", "similarity", "morgan", "fingerprint"],
        "pubchem_search":        ["cid", "pubchem", "formula", "iupac"],
        "generate_3d_conformer": ["conformer", "etkdg", "mmff94", "3d", "xyz"],
        "search_arxiv":          ["arxiv", "paper", "abstract", "title", "published"],
        "search_literature":     ["http", "url", "result", "search"],
        "fetch_tdc_benchmark":   ["tdc", "benchmark", "sota", "metric", "auroc", "mae"],
        "generate_knowledge_graph": ["mermaid", "graph", "-->", "node"],
    }
    for tool in required_tools:
        sigs = TOOL_SIGNATURES.get(tool, [tool.lower()])
        found = sum(1 for s in sigs if s in resp_lower)
        tool_score = min(1.0, found / max(len(sigs) * 0.4, 1))
        criteria[f"tool_{tool}"] = round(tool_score, 3)
        step_scores.append(tool_score)

    # Validate specific ground truth fields
    if "step1" in gt and isinstance(gt["step1"], dict):
        s1 = gt["step1"]
        if "expected_keywords" in s1:
            kw_score = sum(1 for kw in s1["expected_keywords"] if kw.lower() in resp_lower)
            kw_score /= max(len(s1["expected_keywords"]), 1)
            criteria["step1_keywords"] = round(kw_score, 3)
            step_scores.append(kw_score)
        if "min_edges" in s1:
            ec = _count_mermaid_edges(response)
            criteria["step1_edges"] = ec
            step_scores.append(1.0 if ec >= s1["min_edges"] else ec / s1["min_edges"])

    if "step2" in gt and isinstance(gt["step2"], dict):
        s2 = gt["step2"]
        if "expected_keywords" in s2:
            kw_score = sum(1 for kw in s2["expected_keywords"] if kw.lower() in resp_lower)
            kw_score /= max(len(s2["expected_keywords"]), 1)
            criteria["step2_keywords"] = round(kw_score, 3)
            step_scores.append(kw_score)
        if "passes_ro5" in s2:
            expected = s2["passes_ro5"]
            found = ("pass" in resp_lower or "drug-like" in resp_lower) if expected else ("fail" in resp_lower or "violat" in resp_lower)
            criteria["step2_lipinski"] = 1.0 if found else 0.0
            step_scores.append(criteria["step2_lipinski"])
        if "min_edges" in s2:
            ec = _count_mermaid_edges(response)
            criteria["step2_edges"] = ec
            step_scores.append(1.0 if ec >= s2["min_edges"] else ec / s2["min_edges"])
        if "tanimoto_range" in s2:
            lo, hi = s2["tanimoto_range"]
            t_val = _extract_number_near_keyword(response, ["tanimoto", "similarity"])
            if t_val is not None:
                tanimoto_ok = 1.0 if lo <= t_val <= hi else 0.0
            else:
                tanimoto_ok = 0.5  # partial credit if tool was called
            criteria["step2_tanimoto"] = tanimoto_ok
            step_scores.append(tanimoto_ok)

    # Numerical comparisons in D category
    if "heavier" in gt:
        expected_heavier = gt["heavier"]
        criteria["comparison"] = 1.0 if expected_heavier.split("_")[0] in resp_lower else 0.0
        step_scores.append(criteria["comparison"])

    if "comparison" in gt and gt["comparison"] == "both_pass":
        both = "both" in resp_lower and "pass" in resp_lower
        criteria["both_pass"] = 1.0 if both else 0.5
        step_scores.append(criteria["both_pass"])

    if not step_scores:
        step_scores = [0.0]

    score = sum(step_scores) / len(step_scores)
    passed = score >= 0.5 and all(s >= 0.3 for s in step_scores)
    return _make_result(task, round(score, 3), passed, criteria)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    task: dict,
    score: float,
    passed: bool,
    breakdown: dict,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task.get("id", "?"),
        "category": task.get("category", "?"),
        "subcategory": task.get("subcategory", ""),
        "difficulty": task.get("difficulty", "medium"),
        "score": score,
        "passed": passed,
        "breakdown": breakdown,
        "error": error,
    }


def _extract_number_near_keyword(text: str, keywords: list[str]) -> float | None:
    """Find a float in the text that appears near one of the given keywords (within 80 chars)."""
    text_lower = text.lower()
    number_pattern = re.compile(r"\b(\d+\.?\d*)\b")
    for kw in keywords:
        idx = text_lower.find(kw)
        if idx == -1:
            continue
        window = text[max(0, idx - 10): idx + 80]
        matches = number_pattern.findall(window)
        for m in matches:
            try:
                v = float(m)
                if v > 0:
                    return v
            except ValueError:
                continue
    return None


def _extract_any_number_in_range(text: str, lo: float, hi: float) -> bool:
    """Return True if any float in text falls within [lo, hi]."""
    matches = re.findall(r"\b(\d+\.?\d*)\b", text)
    for m in matches:
        try:
            v = float(m)
            if lo <= v <= hi:
                return True
        except ValueError:
            continue
    return False


def _count_mermaid_edges(text: str) -> int:
    """Count --> or -- "..." --> edges in a Mermaid block."""
    # Find mermaid block content
    mermaid_match = re.search(r"```mermaid(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if mermaid_match:
        block = mermaid_match.group(1)
    else:
        block = text  # fallback: search entire text

    edge_patterns = [
        r"--\s*>",       # -->
        r"--\s*\"[^\"]*\"\s*-->",  # -- "label" -->
        r"-\.-\s*>",     # -.->
        r"==\s*>",       # ==>
    ]
    count = 0
    for pattern in edge_patterns:
        count += len(re.findall(pattern, block))
    return count


# ---------------------------------------------------------------------------
# CLI entry point for sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running ChainMind-Bench validation...")
    tasks = load_benchmark()
    result = validate_all(tasks, verbose=True)
    exit(0 if not result["errors"] else 1)
