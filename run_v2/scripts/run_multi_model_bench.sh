#!/usr/bin/env bash
# =============================================================================
# run_multi_model_bench.sh — Full multi-model ChainMind-Bench evaluation
#
# Sequentially evaluates each model from the registry against ChainMind-Bench.
# Each model is started fresh, evaluated, then stopped before the next one.
# This avoids VRAM conflicts on a single RTX 3090 (24GB).
#
# For each model, runs TWO systems:
#   {model}_direct     — Parametric knowledge only (no MCP tools)
#   {model}_chainmind  — Full ChainMind (MCP + A2A tools)
#
# Produces:
#   results/multi_model/                <- Root output directory
#   ├── {model_key}_direct/             <- Per-model direct results
#   ├── {model_key}_chainmind/          <- Per-model ChainMind results
#   ├── multi_model_table1.md           <- Combined Table 1 (TSR across all models)
#   └── multi_model_summary.json        <- Machine-readable summary
#
# Usage:
#   bash scripts/run_multi_model_bench.sh                  # All 5 models, full 100 tasks
#   bash scripts/run_multi_model_bench.sh --models "qwen2.5-7b llama3.1-8b" --n 20
#   bash scripts/run_multi_model_bench.sh --category A    # Cat-A only
#   bash scripts/run_multi_model_bench.sh --dry-run        # Print plan, don't run
#
# Environment:
#   CUDA_VISIBLE_DEVICES  (default: 0)
#   HF_TOKEN              Required for meta-llama models
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_ROOT="$PROJECT_DIR/results/multi_model"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
export HF_HOME="${HF_HOME:-/scratch/nishanth.r/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Load HF_TOKEN from .env if present
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -E '^HF_TOKEN=' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
ALL_MODELS="qwen2.5-7b llama3.1-8b deepseek-r1-7b biomistral-7b phi3.5-mini"
MODELS="${ALL_MODELS}"
CATEGORY="all"
N_TASKS=""
DRY_RUN=false
SKIP_DIRECT=false  # Set true to only run chainmind (saves time)
N_RUNS=1           # Increase to 3 for statistical rigor (mean±std)

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)   MODELS="$2";    shift 2 ;;
        --category) CATEGORY="$2";  shift 2 ;;
        --n)        N_TASKS="$2";   shift 2 ;;
        --runs)     N_RUNS="$2";    shift 2 ;;
        --dry-run)  DRY_RUN=true;   shift ;;
        --chainmind-only) SKIP_DIRECT=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_TASKS_FLAG=""
[ -n "$N_TASKS" ] && N_TASKS_FLAG="--n $N_TASKS"
CAT_FLAG=""
[ "$CATEGORY" != "all" ] && CAT_FLAG="--category $CATEGORY"

# ---------------------------------------------------------------------------
# Print plan
# ---------------------------------------------------------------------------
echo "================================================================"
echo "  ChainMind Multi-Model Benchmark"
echo "================================================================"
echo "  Models   : $MODELS"
echo "  Category : $CATEGORY"
echo "  N Tasks  : ${N_TASKS:-all (100)}"
echo "  N Runs   : $N_RUNS"
echo "  Output   : $RESULTS_ROOT"
echo "================================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run:"
    for MODEL_KEY in $MODELS; do
        echo "  ✦ $MODEL_KEY — direct"
        [ "$SKIP_DIRECT" = false ] && echo "  ✦ $MODEL_KEY — chainmind"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Helper: start model, wait for health, run bench, stop model
# ---------------------------------------------------------------------------

run_bench_for_model() {
    local MODEL_KEY="$1"
    local SYSTEM_KEY="$2"   # {model}_direct or {model}_chainmind
    local RUN_IDX="$3"

    local OUT_DIR="$RESULTS_ROOT/${MODEL_KEY}/run_${RUN_IDX}"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Model: $MODEL_KEY | System: $SYSTEM_KEY | Run: $RUN_IDX"
    echo "──────────────────────────────────────────────────────────"

    # Get port from registry
    PORT=$(python -c "
from chainmind.eval.model_registry import get_model
print(get_model('$MODEL_KEY').port)
")

    # Start model server in background
    bash "$SCRIPT_DIR/start_model_server.sh" "$MODEL_KEY" --background
    echo "  ⏳ Waiting for $MODEL_KEY to become ready on port $PORT..."

    MAX_WAIT=180; WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "  ✅ $MODEL_KEY ready (${WAITED}s)"
            break
        fi
        sleep 4; WAITED=$((WAITED+4))
        echo "     ...waiting (${WAITED}s/${MAX_WAIT}s)"
    done

    if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "  ❌ $MODEL_KEY failed to start. Skipping."
        cat "$LOG_DIR/vllm_${MODEL_KEY}.log" | tail -20 || true
        return 1
    fi

    # Patch settings to point at this model's port
    VLLM_BASE_URL="http://localhost:$PORT/v1" \
    VLLM_PORT="$PORT" \
    python -m chainmind.eval.bench_runner \
        --mode full \
        --system "$SYSTEM_KEY" \
        $N_TASKS_FLAG \
        $CAT_FLAG \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/${SYSTEM_KEY}_run${RUN_IDX}.log"

    # Stop server to free VRAM for next model
    echo "  🛑 Stopping $MODEL_KEY server..."
    pkill -f "vllm.*--port.*$PORT" 2>/dev/null || true
    sleep 5
    echo "  ✅ $MODEL_KEY stopped. VRAM released."
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
FAILED_MODELS=()
COMPLETED_MODELS=()

for MODEL_KEY in $MODELS; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  START: $MODEL_KEY"
    echo "════════════════════════════════════════════════════════"

    for RUN_IDX in $(seq 1 $N_RUNS); do
        # Direct baseline first
        if [ "$SKIP_DIRECT" = false ]; then
            if ! run_bench_for_model "$MODEL_KEY" "${MODEL_KEY}_direct" "$RUN_IDX"; then
                FAILED_MODELS+=("${MODEL_KEY}_direct_run${RUN_IDX}")
                continue
            fi
        fi

        # ChainMind system
        if ! run_bench_for_model "$MODEL_KEY" "${MODEL_KEY}_chainmind" "$RUN_IDX"; then
            FAILED_MODELS+=("${MODEL_KEY}_chainmind_run${RUN_IDX}")
            continue
        fi
    done

    COMPLETED_MODELS+=("$MODEL_KEY")
    echo ""
    echo "  ✅ $MODEL_KEY — all runs complete"
done

# ---------------------------------------------------------------------------
# Aggregate results → Combined Table 1
# ---------------------------------------------------------------------------
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Aggregating results → Combined Paper Tables"
echo "════════════════════════════════════════════════════════"

python - <<'PYEOF'
import json, pathlib, datetime
from collections import defaultdict

results_root = pathlib.Path("results/multi_model")
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Collect all result files
all_records = defaultdict(list)  # system_key → list of result dicts

for json_file in sorted(results_root.rglob("*.json")):
    try:
        data = json.loads(json_file.read_text())
        for r in data.get("results", []):
            key = r.get("system", "unknown")
            all_records[key].append(r)
    except Exception as e:
        print(f"  [WARN] Couldn't parse {json_file}: {e}")

if not all_records:
    print("  [WARN] No results found yet.")
    exit(0)

cats = ["A", "B", "C", "D"]

def compute_tsr(records, cat=None):
    recs = [r for r in records if r.get("category") == cat] if cat else records
    if not recs: return "N/A"
    passed = sum(1 for r in recs if r.get("passed"))
    return f"{passed/len(recs)*100:.1f}"

def compute_tool_rate(records):
    tool_recs = [r for r in records if r.get("n_tools_called", 0) > 0]
    return f"{len(tool_recs)/max(len(records),1)*100:.1f}"

def compute_avg_lat(records):
    lats = [r["latency_ms"] for r in records if "latency_ms" in r]
    return f"{sum(lats)/len(lats):.0f}" if lats else "N/A"

# Build table
lines = [
    f"# Multi-Model Benchmark — Table 1: Task Success Rate (TSR %)",
    f"Generated: {ts}\n",
    f"| {'System':<38} | {'Cat-A':>6} | {'Cat-B':>6} | {'Cat-C':>6} | {'Cat-D':>6} | {'Overall':>7} | {'Tool%':>6} | {'Lat(ms)':>8} | N |",
    f"| {'-'*38} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*7} | {'-'*6} | {'-'*8} | - |",
]

summary = {}
for sys_name in sorted(all_records.keys()):
    recs = all_records[sys_name]
    row = {
        "n": len(recs),
        "overall_tsr": compute_tsr(recs),
        "tool_rate": compute_tool_rate(recs),
        "avg_lat_ms": compute_avg_lat(recs),
    }
    for c in cats:
        row[f"cat_{c}_tsr"] = compute_tsr(recs, c)
    summary[sys_name] = row

    lines.append(
        f"| {sys_name:<38} | {row['cat_A_tsr']:>6} | {row['cat_B_tsr']:>6} "
        f"| {row['cat_C_tsr']:>6} | {row['cat_D_tsr']:>6} | {row['overall_tsr']:>7} "
        f"| {row['tool_rate']:>6} | {row['avg_lat_ms']:>8} | {row['n']} |"
    )

lines += [
    "",
    "> **Higher is better for TSR%. Tool% shows tasks where ≥1 MCP tool was invoked.**",
    "",
    "## Key findings",
    "- [Populate after reviewing results]",
]

report = "\n".join(lines)
out_md  = results_root / f"multi_model_table1_{ts}.md"
out_json = results_root / f"multi_model_summary_{ts}.json"

out_md.write_text(report)
out_json.write_text(json.dumps(summary, indent=2))

print(report)
print(f"\n💾 Table saved → {out_md}")
print(f"💾 Summary saved → {out_json}")
PYEOF

echo ""
echo "================================================================"
echo "  MULTI-MODEL BENCHMARK COMPLETE"
if [ ${#COMPLETED_MODELS[@]} -gt 0 ]; then
    echo "  ✅ Completed: ${COMPLETED_MODELS[*]}"
fi
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "  ❌ Failed: ${FAILED_MODELS[*]}"
fi
echo ""
echo "  Next steps:"
echo "  1. Review results/multi_model/multi_model_table1_*.md"
echo "  2. Run 3x for statistical rigor:"
echo "     bash scripts/run_multi_model_bench.sh --runs 3 --n 20"
echo "  3. Run ablation on best model:"
echo "     python -m chainmind.eval.ablation --mode full --n 40"
echo "  4. Generate paper figures:"
echo "     python scripts/plot_results.py --input results/multi_model/"
echo "================================================================"
