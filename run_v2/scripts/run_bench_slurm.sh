#!/usr/bin/env bash
# =============================================================================
# run_bench_slurm.sh — Submit ChainMind benchmark as a new SLURM job
#
# ROOT CAUSE:
#   Current SLURM job (ID: $SLURM_JOB_ID) has only 2GB RAM allocated.
#   Qwen-7B weights = 14.25 GiB. OOM-kill occurs during weight loading.
#   The GPU itself (RTX 3090, 24 GB VRAM) is fine — it's the CPU RAM limit.
#
# SOLUTION:
#   Submit a new job with --mem=32G to give vLLM enough RAM for weight loading
#   while keeping the same GPU.
#
# Usage:
#   bash scripts/run_bench_slurm.sh
#   bash scripts/run_bench_slurm.sh --model qwen2.5-7b --category A --n 20
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
CATEGORY="${CATEGORY:-A}"
N_TASKS="${N_TASKS:-}"
MODE="${MODE:-full}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL_KEY="$2";  shift 2 ;;
        --category) CATEGORY="$2";  shift 2 ;;
        --n)        N_TASKS="$2";   shift 2 ;;
        --mode)     MODE="$2";      shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

N_FLAG=""
[ -n "$N_TASKS" ] && N_FLAG="--n $N_TASKS"

if [[ "$MODEL_KEY" == *"qwen"* ]]; then
    SYS_CHAINMIND="chainmind_qwen"
    SYS_DIRECT="qwen_direct"
else
    SYS_CHAINMIND="chainmind_gpt4"
    SYS_DIRECT="gpt4_direct"
fi

echo "================================================================"
echo "  ChainMind Benchmark — SLURM Submission"
echo "  Current RAM limit  : $(cat /proc/$$/status | grep VmPeak || echo 'unknown')"
echo "  Submitting with    : --mem=32G --gres=gpu:1 --cpus-per-task=8"
echo "  Model              : $MODEL_KEY"
echo "  Category           : $CATEGORY"
echo "================================================================"

# Create the job script
JOB_SCRIPT=$(mktemp --suffix=".sh")
cat > "$JOB_SCRIPT" << SLURM_SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=chainmind-bench
#SBATCH --output=${PROJECT_DIR}/logs/bench_%j.log
#SBATCH --error=${PROJECT_DIR}/logs/bench_%j.err
#SBATCH --partition=plafnet2
#SBATCH --account=plafnet2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

cd "$PROJECT_DIR"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/scratch/nishanth.r/hf_cache"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export ENVIRONMENT="development" # Override SLURM's injected "BATCH" value

echo "=== SLURM Job Started: \$SLURM_JOB_ID ==="
echo "  RAM limit: \$(grep MemTotal /proc/meminfo || echo unknown)"
echo "  GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i 0)"
echo ""

# Start vLLM
echo "Starting vLLM..."
bash "$SCRIPT_DIR/start_vllm_optimized.sh" &
VLLM_PID=\$!

# Wait for server
echo "Waiting for vLLM (up to 300s)..."
MAX_WAIT=300; WAITED=0
while [ \$WAITED -lt \$MAX_WAIT ]; do
    if curl -sf "http://localhost:8100/health" >/dev/null 2>&1; then
        echo "✅ vLLM ready in \${WAITED}s"
        break
    fi
    sleep 5; WAITED=\$((WAITED+5))
    (( \$WAITED % 30 == 0 )) && echo "  ...(\${WAITED}s)"
done

if ! curl -sf "http://localhost:8100/health" >/dev/null 2>&1; then
    echo "❌ vLLM failed to start"
    kill \$VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run benchmark
echo ""
echo "=== Running ChainMind-Bench ==="
python -m chainmind.eval.bench_runner \
    --mode $MODE \
    --system $SYS_CHAINMIND \
    --category $CATEGORY \
    $N_FLAG \
    --output-dir "$PROJECT_DIR/results/bench" \
    2>&1

echo ""
echo "=== Running baseline (direct, no tools) ==="
python -m chainmind.eval.bench_runner \
    --mode $MODE \
    --system $SYS_DIRECT \
    --category $CATEGORY \
    $N_FLAG \
    --output-dir "$PROJECT_DIR/results/bench" \
    2>&1

kill \$VLLM_PID 2>/dev/null || true
echo "=== Benchmark Complete ==="
SLURM_SCRIPT

chmod +x "$JOB_SCRIPT"
echo ""
echo "Job script: $JOB_SCRIPT"

# Check if sbatch is available
if command -v sbatch &>/dev/null; then
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo ""
    echo "✅ Submitted SLURM job: $JOB_ID"
    echo "   Monitor: squeue -j $JOB_ID"
    echo "   Logs   : tail -f $PROJECT_DIR/logs/bench_${JOB_ID}.log"
    echo "   Cancel : scancel $JOB_ID"
else
    echo ""
    echo "⚠️  sbatch not found. Running directly..."
    echo "   (This will OOM-kill with current 2GB RAM limit)"
    echo "   Submit via: sbatch $JOB_SCRIPT"
fi
