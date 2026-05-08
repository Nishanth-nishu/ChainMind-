#!/usr/bin/env bash
# =============================================================================
# run_exp1_hard_tasks.sh — Run Category D Hard Reasoning Benchmark
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXP_DIR="$PROJECT_DIR/results/experiments/exp1_hard_tasks/bench"
MODEL_KEY="${1:-qwen2.5-7b}"

mkdir -p "$EXP_DIR"

if [[ "$MODEL_KEY" == *"qwen"* ]]; then
    SYS_CHAINMIND="chainmind_qwen"
    SYS_DIRECT="qwen_direct"
else
    SYS_CHAINMIND="chainmind_gpt4"
    SYS_DIRECT="gpt4_direct"
fi

echo "=== Running Hard Tasks (Cat D) Baseline ==="
python -m chainmind.eval.bench_runner \
    --mode full \
    --system $SYS_DIRECT \
    --category D \
    --output-dir "$EXP_DIR" 

echo "=== Running Hard Tasks (Cat D) Agentic ==="
python -m chainmind.eval.bench_runner \
    --mode full \
    --system $SYS_CHAINMIND \
    --category D \
    --output-dir "$EXP_DIR"
