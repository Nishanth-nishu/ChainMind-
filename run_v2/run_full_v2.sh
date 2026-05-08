#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J CHAINMIND_V2
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode118
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/nishanth.r/sys_elvle_ai/run_v2/slurm_logs/run_v2_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

echo "=========================================="
echo "SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "START TIME     = $(date)"
echo "=========================================="

# Strictly move to the NEW workspace
PROJECT_DIR="/scratch/nishanth.r/sys_elvle_ai/run_v2"
cd "$PROJECT_DIR" || exit 1
echo "Working directory: $(pwd)"

# Create local results folder
mkdir -p results slurm_logs

# Force Scratch usage for all caches
export HF_HOME="/scratch/nishanth.r/.home_cache_migration/huggingface"
export PIP_CACHE_DIR="/scratch/nishanth.r/.home_cache_migration/pip"
export PYTHONPATH=".:$PYTHONPATH"
export ENVIRONMENT="development"

# Use the existing venv (which is on scratch)
source ../.venv/bin/activate
echo "Using environment:"
which python

# Start vLLM in background (Optimized for research runs)
echo "Starting vLLM..."
bash scripts/start_vllm_optimized.sh &
VLLM_PID=$!

# Wait for server readiness
echo "Waiting for vLLM..."
MAX_WAIT=300; WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:8100/health" >/dev/null 2>&1; then
        echo "✅ vLLM ready in ${WAITED}s"
        break
    fi
    sleep 5; WAITED=$((WAITED+5))
done

if ! curl -sf "http://localhost:8100/health" >/dev/null 2>&1; then
    echo "❌ vLLM failure. Aborting."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run everything from scratch (1-8)
echo "Executing Full Experiment Suite (1 to 8)..."
bash scripts/run_all_experiments.sh --full --start-from 1

echo "=========================================="
echo "RUN COMPLETED"
echo "END TIME = $(date)"
echo "=========================================="

# Cleanup
kill $VLLM_PID 2>/dev/null
