#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J CHAINMIND_EXPS
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode118
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/nishanth.r/sys_elvle_ai/slurm_logs/chainmind_exps_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

echo "=========================================="
echo "SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "START TIME     = $(date)"
echo "=========================================="

# 1. Move to the workspace
PROJECT_DIR="/scratch/nishanth.r/sys_elvle_ai"
cd "$PROJECT_DIR" || exit 1
echo "Working directory: $(pwd)"

# 2. Setup the "new folder" for SLURM stdout logs
mkdir -p slurm_logs

# 3. Proper python environment initialization
source .venv/bin/activate
echo "Activated Conda/Venv environment:"
which python
python --version

export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/scratch/nishanth.r/hf_cache"

# 4. Start vLLM Background Server for the local models
echo "Starting vLLM..."
bash scripts/start_vllm_optimized.sh &
VLLM_PID=$!

echo "Waiting for vLLM (up to 300s)..."
MAX_WAIT=300; WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:8100/health" >/dev/null 2>&1; then
        echo "✅ vLLM ready in ${WAITED}s"
        break
    fi
    sleep 5; WAITED=$((WAITED+5))
    (( $WAITED % 30 == 0 )) && echo "  ...(${WAITED}s)"
done

if ! curl -sf "http://localhost:8100/health" >/dev/null 2>&1; then
    echo "❌ vLLM failed to start. Aborting job."
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# 5. Run all experiments from 1 to 8 recursively
echo "Starting full ChainMind Experiment Suite..."
bash scripts/run_all_experiments.sh --full --start-from 1

echo "=========================================="
echo "EXPERIMENTS COMPLETED"
echo "END TIME = $(date)"
echo "=========================================="

kill $VLLM_PID 2>/dev/null || true
