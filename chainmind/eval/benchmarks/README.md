# ChainMind-Bench: Dataset Card

## Overview

**ChainMind-Bench** is a curated evaluation benchmark for agentic drug discovery systems. It contains **100 tasks** designed to measure Task Success Rate (TSR) across four functionally distinct categories.

All ground truths are **deterministically computed** — either directly via RDKit/PubChem (Category A) or via verifiable keyword matching and structural rules (Categories B–D). No human annotation is needed for grading.

---

## Task Distribution

| Category | Name | Count | Scorer | Ground Truth Source |
|----------|------|------:|--------|---------------------|
| **A** | Molecular Property | 40 | Numerical tolerance (±5%) | RDKit / PubChem REST API |
| **B** | Literature Retrieval | 30 | Keyword recall (≥ 60%) | ArXiv API / TDC leaderboard |
| **C** | Knowledge Graph | 15 | Mermaid validity + edge count ≥ 5 | Structural parse |
| **D** | Multi-Step Chain | 15 | Composite (per-step subscores) | Combination of A + B + C |
| **Total** | | **100** | | |

---

## Category A — Molecular Property (40 tasks)

Subcategories:
- **Lipinski** (16 tasks): Does molecule X pass Lipinski's Rule of 5? Ground truth: RDKit `Descriptors.MolWt`, `MolLogP`, `CalcNumLipinskiHBD/HBA`. Tolerance ±5%.
- **Molecular Weight** (5 tasks): Exact MW computation.
- **Tanimoto Similarity** (6 tasks): Morgan fingerprint (radius=2, 2048 bits) similarity. Answer must fall within a ±0.1 range.
- **PubChem Lookup** (7 tasks): CID, formula, weight from PubChem REST API.
- **3D Conformer** (6 tasks): ETKDG + MMFF94 generation. Pass = converged + correct formula.

Scoring: numerical match with tolerance. Boolean answers matched by keyword presence.

---

## Category B — Literature Retrieval (30 tasks)

Subcategories:
- **ArXiv Search** (16 tasks): Find recent papers on a topic. Scored by keyword recall ≥ 0.6 across 5 expected concepts.
- **TDC Benchmark** (6 tasks): Retrieve cached ADMET benchmark info. Scored by keyword recall.
- **Web Search** (8 tasks): DuckDuckGo search for drug discovery concepts. Keyword recall ≥ 0.6.

Bonus: +0.10 if tool call evidence (URL, abstract, arxiv_id) is present in response.

---

## Category C — Knowledge Graph (15 tasks)

Requirements for passing:
1. Response must contain a ` ```mermaid ` block.
2. ≥ 5 directed edges (`-->`) in the graph.
3. ≥ 40% of required concepts mentioned in the response.

Weighted score: 35% (mermaid block) + 40% (edge count) + 25% (concept recall).

---

## Category D — Multi-Step Chain (15 tasks)

Composite tasks requiring ≥ 2 tools in sequence. Subcategories:
- **Search → Analyze**: PubChem lookup → Lipinski assessment
- **Analyze → Compare**: Two Lipinski calls → comparison
- **Research → Explain**: ArXiv search → KG generation
- **TDC → Research**: Benchmark lookup → ArXiv search

Scoring: mean of per-step subscores. Passes if all steps ≥ 0.3 and mean ≥ 0.5.

---

## Evaluation Protocol

```bash
# 1. Validate benchmark loads correctly (no LLM calls)
python chainmind/eval/benchmarks/ground_truth_validator.py

# 2. Run a 5-task sample against ChainMind (Qwen-7B)
python -m chainmind.eval.bench_runner --mode sample --n 5 --system chainmind_qwen

# 3. Full benchmark evaluation
python -m chainmind.eval.bench_runner --mode full --system chainmind_qwen

# 4. Ablation study (Table 4)
python -m chainmind.eval.ablation --ablation all --mode sample --n 20

# 5. Generate paper tables from saved results
python -m chainmind.eval.bench_runner --mode report --results-file results/bench/bench_full_*.json
```

---

## Reproducibility

- All Category A computations use RDKit with `randomSeed=42` for 3D conformers.
- The benchmark JSON is frozen at `chainmind/eval/benchmarks/chainmind_bench.json`.
- Validation pass thresholds are documented in `ground_truth_validator.py`.
- No proprietary data: all ground truths sourced from open APIs (PubChem, ArXiv, TDC).

---

## Citation

When using ChainMind-Bench, please cite:

```bibtex
@misc{chainmind2026,
  title   = {ChainMind: Standards-Based Multi-Agent Drug Discovery with Local LLMs via MCP and A2A Protocols},
  author  = {Nishanth R.},
  year    = {2026},
  note    = {Preprint, under review},
}
```
