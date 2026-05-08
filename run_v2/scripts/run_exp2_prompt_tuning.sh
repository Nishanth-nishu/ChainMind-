#!/usr/bin/env bash
# =============================================================================
# run_exp2_prompt_tuning.sh — Run Category A with Optimized Prompt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXP_DIR="$PROJECT_DIR/results/experiments/exp2_prompt_tuning/bench"
MODEL_KEY="${1:-qwen2.5-7b}"

mkdir -p "$EXP_DIR"

if [[ "$MODEL_KEY" == *"qwen"* ]]; then
    SYS_CHAINMIND="chainmind_qwen"
    SYS_DIRECT="qwen_direct"
else
    SYS_CHAINMIND="chainmind_gpt4"
    SYS_DIRECT="gpt4_direct"
fi

echo "=== Running Prompt Tuning (Cat A) Baseline ==="
python -m chainmind.eval.bench_runner \
    --mode full \
    --system $SYS_DIRECT \
    --category A \
    --output-dir "$EXP_DIR" 

echo "=== Running Prompt Tuning (Cat A) Agentic ==="
python -m chainmind.eval.bench_runner \
    --mode full \
    --system $SYS_CHAINMIND \
    --category A \
    --output-dir "$EXP_DIR"
