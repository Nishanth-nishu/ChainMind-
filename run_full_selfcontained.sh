#!/usr/bin/env bash
# =============================================================================
# run_full_selfcontained.sh
# ChainMind Benchmark — Self-Contained SLURM Script
# Based on bash_jepa_118.sh template (gnode118, plafnet2, GPU:1)
#
# This script is FULLY SELF-CONTAINED:
#   1. Checks/downloads the LLM model weights if missing
#   2. Starts the vLLM server
#   3. Waits for readiness
#   4. Runs ALL 8 experiments (100 tasks each)
#   5. Saves results + logs to run_v3/ subdirectory on scratch
#
# Submit:  sbatch run_full_selfcontained.sh
# Monitor: tail -f /scratch/nishanth.r/sys_elvle_ai/run_v3/slurm_logs/run_v3_<JOB_ID>.log
# =============================================================================
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J CHAINMIND_V3
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode118
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/nishanth.r/sys_elvle_ai/run_v3/slurm_logs/run_v3_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

set -euo pipefail

echo "=========================================="
echo "SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "SLURM_NODELIST  = $SLURM_NODELIST"
echo "START TIME      = $(date)"
echo "=========================================="

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/nishanth.r/sys_elvle_ai"
RUN_DIR="${PROJECT_DIR}/run_v3"
MODEL_REPO="Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE="/scratch/nishanth.r/hf_cache"
MODEL_CACHE_DIR="${MODEL_CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct"
VENV="${PROJECT_DIR}/.venv"

# ── Create output structure ────────────────────────────────────────────────
mkdir -p "${RUN_DIR}/slurm_logs"
mkdir -p "${RUN_DIR}/results"
mkdir -p "${RUN_DIR}/logs"

# ── Environment variables ─────────────────────────────────────────────────
export HF_HOME="${MODEL_CACHE}"
export PIP_CACHE_DIR="/scratch/nishanth.r/.home_cache_migration/pip"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export ENVIRONMENT="development"   # Prevent Pydantic rejecting SLURM's "BATCH" value
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# ── Activate virtual environment ───────────────────────────────────────────
cd "$PROJECT_DIR"
source "${VENV}/bin/activate"
echo "Python: $(which python) | $(python --version)"

# ── Sanity check ──────────────────────────────────────────────────────────
python - <<EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF

# ── MODEL DOWNLOAD CHECK (runs in parallel with env setup) ─────────────────
echo ""
echo "=========================================="
echo "  Checking LLM model weights..."
echo "=========================================="

if [ -d "${MODEL_CACHE_DIR}/snapshots" ] && [ "$(du -sm ${MODEL_CACHE_DIR} | cut -f1)" -gt 10000 ]; then
    echo "✅ Model weights found at ${MODEL_CACHE_DIR} ($(du -sh ${MODEL_CACHE_DIR} | cut -f1))"
else
    echo "⚠️  Model weights not found or incomplete. Downloading ${MODEL_REPO}..."
    echo "    This takes ~5 minutes on a fast connection. Download runs in background."
    python - <<PYEOF
import os
os.environ["HF_HOME"] = "${MODEL_CACHE}"
from huggingface_hub import snapshot_download
print(f"Downloading {MODEL_REPO}...")
path = snapshot_download(
    repo_id="${MODEL_REPO}",
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    local_dir_use_symlinks=False,
)
print(f"Download complete: {path}")
PYEOF
    echo "✅ Model download complete."
fi

# ── START vLLM ────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Starting vLLM server..."
echo "=========================================="

# Kill any stale server on port 8100
if lsof -ti:8100 > /dev/null 2>&1; then
    echo "Port 8100 in use — killing stale process..."
    kill "$(lsof -ti:8100)" 2>/dev/null || true
    sleep 3
fi

bash "${PROJECT_DIR}/scripts/start_vllm_optimized.sh" &
VLLM_PID=$!

echo "vLLM PID: ${VLLM_PID}"
echo "Waiting for model to load (cold start = 2-5 min)..."

MAX_WAIT=360; WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:8100/v1/models" > /dev/null 2>&1; then
        echo "✅ vLLM ready in ${WAITED}s"
        break
    fi
    sleep 5; WAITED=$((WAITED + 5))
    (( WAITED % 30 == 0 )) && echo "  ...(${WAITED}s elapsed)"
done

if ! curl -sf "http://localhost:8100/v1/models" > /dev/null 2>&1; then
    echo "❌ vLLM failed to start after ${MAX_WAIT}s. Aborting."
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Show which model is loaded
echo "Loaded model: $(curl -s http://localhost:8100/v1/models | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])")"

# ── LINK results into run_v3 so the runner writes there ───────────────────
# The run_all_experiments.sh writes to results/experiments/ relative to the
# project root. We redirect via symlink.
rm -f "${PROJECT_DIR}/results_current"
ln -sf "${RUN_DIR}/results" "${PROJECT_DIR}/results_current"

# ── RUN ALL 8 EXPERIMENTS ─────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Running Experiments 1-8 (100 tasks each)"
echo "=========================================="

bash "${PROJECT_DIR}/scripts/run_all_experiments.sh" --full --start-from 1 \
    2>&1 | tee "${RUN_DIR}/logs/full_run_$(date +%Y%m%d_%H%M%S).log"

# ── DONE ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  ALL EXPERIMENTS COMPLETED"
echo "  Results: ${RUN_DIR}/results/"
echo "  Logs:    ${RUN_DIR}/logs/"
echo "  END TIME = $(date)"
echo "=========================================="

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "vLLM shut down cleanly."
