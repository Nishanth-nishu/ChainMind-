#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — Run all 8 ChainMind experiments
#
# Usage:
#   bash scripts/run_all_experiments.sh            # sample mode, N=20 per exp
#   bash scripts/run_all_experiments.sh --full     # full 100-task benchmark
#   bash scripts/run_all_experiments.sh --n 10     # custom sample size
#   bash scripts/run_all_experiments.sh --exp 4    # single experiment (1-8)
#
# Prereq: vLLM must be running
#   bash scripts/start_vllm.sh
# =============================================================================
set -euo pipefail

# Defaults
MODE="sample"
N_TASKS=20
SINGLE_EXP=""
START_FROM=1
VENV_PATH=".venv"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --full)  MODE="full"; N_TASKS=100 ;;
    --n)     N_TASKS="$2"; shift ;;
    --exp)   SINGLE_EXP="$2"; shift ;;
    --start-from) START_FROM="$2"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
  shift
done

# Activate venv
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPERIMENTS=(
  "1:exp001_reflexion"
  "2:exp002_self_consistency"
  "3:exp003_cove"
  "4:exp004_few_shot"
  "5:exp005_tool_rag"
  "6:exp006_structured_output"
  "7:exp007_chem_rag"
  "8:exp008_debate"
)

LOG_DIR="logs/experiments"
mkdir -p "$LOG_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        ChainMind Research Experiment Suite               ║"
echo "║        Mode: ${MODE} | N: ${N_TASKS} tasks per exp             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PASSED=0
FAILED=0
FAILED_EXPS=()
START_ALL=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
  NUM="${entry%%:*}"
  EXP="${entry##*:}"

  # Skip if running a single experiment
  if [[ -n "$SINGLE_EXP" && "$NUM" != "$SINGLE_EXP" ]]; then
    continue
  fi

  # Skip if resuming and we haven't reached the start group
  if [[ "$NUM" -lt "$START_FROM" ]]; then
    continue
  fi

  LOG_FILE="${LOG_DIR}/${EXP}_$(date +%Y%m%d_%H%M%S).log"

  echo "──────────────────────────────────────────────────────────"
  echo "  [${NUM}/8] ${EXP}"
  echo "  Log: ${LOG_FILE}"
  echo ""

  START=$(date +%s)
  set +e
  python "experiments/${EXP}/run.py" \
    --mode "$MODE" \
    --n "$N_TASKS" \
    2>&1 | tee "$LOG_FILE"
  EXIT_CODE=$?
  set -e
  END=$(date +%s)

  ELAPSED=$((END - START))
  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  ✅  ${EXP} PASSED in ${ELAPSED}s"
    PASSED=$((PASSED + 1))
  else
    echo "  ❌  ${EXP} FAILED (exit ${EXIT_CODE}) in ${ELAPSED}s"
    FAILED=$((FAILED + 1))
    FAILED_EXPS+=("${EXP}")
  fi
  echo ""
done

END_ALL=$(date +%s)
TOTAL=$((END_ALL - START_ALL))

echo "══════════════════════════════════════════════════════════"
echo "  Completed: ${PASSED} passed, ${FAILED} failed (${TOTAL}s total)"
if [[ ${#FAILED_EXPS[@]} -gt 0 ]]; then
  echo "  Failed: ${FAILED_EXPS[*]}"
fi
echo ""
echo "  Comparing results..."
echo ""
python scripts/compare_experiments.py
echo ""
echo "  Markdown table:"
python scripts/compare_experiments.py --format markdown
echo "══════════════════════════════════════════════════════════"
