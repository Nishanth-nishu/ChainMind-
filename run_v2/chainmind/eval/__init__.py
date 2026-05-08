"""
ChainMind Evaluation Framework — Research-grade evaluation suite for D4.

Modules
-------
dataset.py          Original Q&A eval dataset (55 questions, supply chain + D4)
metrics.py          Scoring functions (keyword, judge LLM, latency)
runner.py           Unified multi-mode eval runner (quality / perf / RAG / A/B)
performance.py      vLLM throughput benchmarking (TTFT, TPS, concurrency)
rag_eval.py         RAG faithfulness and retrieval quality evaluation
prompt_ab.py        Prompt variant A/B testing with statistical significance

benchmarks/
  chainmind_bench.json           100-task D4 benchmark (ChainMind-Bench v1.0)
  ground_truth_validator.py      Deterministic scorer for all 4 categories
  README.md                      Dataset card (task distribution, protocol)

bench_runner.py     5-baseline harness → Tables 1, 2, 3, 5 (paper)
ablation.py         4-ablation study   → Table 4 (paper)
"""
