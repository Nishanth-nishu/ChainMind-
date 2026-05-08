#!/usr/bin/env bash
# =============================================================================
# EXP010: Fine-Tune + Evaluate — Complete Pipeline
#
# This single SLURM job does everything sequentially:
#   Phase 1: Build SFT dataset from run_v5 benchmark traces + Mol-Instructions
#   Phase 2: QLoRA SFT (fine-tune Qwen2.5-7B on ChainMind domain data)
#   Phase 3: Merge LoRA adapter into full model
#   Phase 4: Serve fine-tuned model with vLLM on port 8101
#   Phase 5: Evaluate fine-tuned model on ChainMind benchmark (all 100 tasks)
#   Phase 6: Print comparison table: base model vs fine-tuned
#
# GPU Plan:
#   - Phase 1-2:   Training uses full 24GB VRAM (vLLM OFF during training)
#   - Phase 3:     CPU-only (merge adapter)
#   - Phase 4-6:   Inference uses ~14GB VRAM (vLLM serving fine-tuned model)
#
# Wall time estimate: ~6-8h total
#   - Dataset build:  ~20 min
#   - QLoRA training: ~3-5h  (2 epochs, 5K examples, seq_len=2048)
#   - Merge + serve:  ~15 min
#   - Evaluation:     ~1-2h  (100 tasks × ~30-60s each)
#
# Submit: sbatch experiments/exp010_qlora_sft/finetune_and_eval.sh
# Monitor: tail -f logs/exp010_ft_eval_<JOB_ID>.log
# =============================================================================
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J CM_FT_EVAL
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/nishanth.r/sys_elvle_ai/logs/exp010_ft_eval_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   EXP010: QLoRA Fine-Tune + Evaluation Pipeline         ║"
echo "║   Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST           ║"
echo "║   Start: $(date)                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT="/scratch/nishanth.r/sys_elvle_ai"
VENV="${PROJECT}/.venv"
HF_CACHE="/scratch/nishanth.r/hf_cache"
MODEL_BASE="Qwen/Qwen2.5-7B-Instruct"
MODEL_OUT="${PROJECT}/models/chainmind-ft-v1"
LOG_DIR="${PROJECT}/logs"
EVAL_OUT="${PROJECT}/run_ft_eval"
BASELINE="${PROJECT}/run_v5/results"

export HF_HOME="${HF_CACHE}"
export PIP_CACHE_DIR="/scratch/nishanth.r/.home_cache_migration/pip"
export PYTHONPATH="${PROJECT}"
export ENVIRONMENT="development"
export CUDA_VISIBLE_DEVICES=0
export FLASH_ATTENTION_FORCE_BUILD=1

mkdir -p "${LOG_DIR}" "${MODEL_OUT}" "${EVAL_OUT}/results" "${EVAL_OUT}/logs"
cd "$PROJECT"
source "${VENV}/bin/activate"

echo ""
echo "Python: $(python --version) | $(which python)"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 0: Kill any running vLLM to free GPU for training
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 0: Freeing GPU ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for PORT in 8100 8101; do
    if lsof -ti:$PORT > /dev/null 2>&1; then
        echo "  Killing process on port $PORT..."
        kill $(lsof -ti:$PORT) 2>/dev/null || true
        sleep 3
    fi
done
echo "  GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MB available"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Install training dependencies
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 1: Training Dependencies ━━━━━━━━━━━━━━━━━━━━━━━━━"

install_if_missing() {
    python3 -c "import ${1//-/_}" 2>/dev/null && echo "  ✅ $1" && return
    echo "  Installing $1..."
    eval "${2:-pip install $1}" 2>&1 | tail -2
}

install_if_missing bitsandbytes "pip install bitsandbytes --upgrade"
install_if_missing trl "pip install 'trl>=0.12.0'"
install_if_missing accelerate "pip install accelerate"
install_if_missing peft "pip install peft"
install_if_missing datasets "pip install datasets"

if ! python3 -c "import unsloth" 2>/dev/null; then
    echo "  Installing unsloth..."
    pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git" \
        --no-deps 2>&1 | tail -3
    pip install xformers --no-deps 2>&1 | tail -2
fi
echo "  ✅ unsloth"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Build SFT Dataset
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 2: Building SFT Dataset ━━━━━━━━━━━━━━━━━━━━━━━━━━"
SFT_DATA="${PROJECT}/data/sft_dataset.jsonl"

if [ -f "${SFT_DATA}" ] && [ $(wc -l < "${SFT_DATA}") -ge 500 ]; then
    echo "  ✅ SFT dataset exists: $(wc -l < ${SFT_DATA}) examples (skipping rebuild)"
else
    echo "  Building dataset from run_v5 traces + Mol-Instructions..."
    python3 scripts/build_sft_dataset.py \
        --results-dir "${BASELINE}" \
        --max-mol-instructions 5000 \
        2>&1 | tee "${LOG_DIR}/phase2_dataset_${SLURM_JOB_ID}.log"
fi

N_EXAMPLES=$(wc -l < "${SFT_DATA}" 2>/dev/null || echo 0)
echo "  Dataset size: ${N_EXAMPLES} examples"

if [ "${N_EXAMPLES}" -lt 50 ]; then
    echo "  ⚠️  Dataset small (<50). Using synthetic KG examples only — training will still run."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: QLoRA SFT Training
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 3: QLoRA SFT Training ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Model: ${MODEL_BASE}"
echo "  Output: ${MODEL_OUT}"
echo "  Config: r=64, alpha=128, 4-bit NF4, lr=2e-4, 2 epochs"
echo "  Data: ${N_EXAMPLES} examples, max_seq_len=2048"
echo "  Expected: 3-5h training time"
echo ""

TRAIN_START=$(date +%s)

python3 experiments/exp010_qlora_sft/train.py \
    --output-dir "${MODEL_OUT}" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 2048 \
    --learning-rate 2e-4 \
    --epochs 2 \
    2>&1 | tee "${LOG_DIR}/phase3_train_${SLURM_JOB_ID}.log"

TRAIN_ELAPSED=$(( $(date +%s) - TRAIN_START ))
echo ""
echo "  ✅ Training complete in $(( TRAIN_ELAPSED / 3600 ))h $(( (TRAIN_ELAPSED % 3600) / 60 ))m"

# Validate adapter was saved
if [ ! -f "${MODEL_OUT}/adapter/adapter_model.safetensors" ]; then
    echo "  ❌ Adapter not found. Training failed. Check phase3 log."
    exit 1
fi
echo "  Adapter size: $(du -sh ${MODEL_OUT}/adapter/ | cut -f1)"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Serve fine-tuned model with vLLM on port 8101
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 4: Serving Fine-Tuned Model ━━━━━━━━━━━━━━━━━━━━━━"

FT_MODEL="${MODEL_OUT}/merged"
if [ ! -d "${FT_MODEL}" ]; then
    echo "  Merged model not found — adapter-only mode (slower inference)"
    FT_MODEL="${MODEL_OUT}/adapter"
fi

echo "  Model: ${FT_MODEL}"
echo "  Port: 8101 (base model stays off — full GPU for fine-tuned)"

# Start vLLM for fine-tuned model
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${FT_MODEL}" \
    --served-model-name "chainmind-ft" \
    --port 8101 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --dtype float16 \
    --disable-log-requests \
    > "${LOG_DIR}/phase4_vllm_ft_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_FT_PID=$!
echo "  vLLM PID: ${VLLM_FT_PID}"

# Wait for vLLM to be ready (up to 5 min)
echo "  Waiting for vLLM to be ready..."
WAIT=0
until curl -sf http://0.0.0.0:8101/health > /dev/null 2>&1; do
    sleep 10; WAIT=$((WAIT + 10))
    if [ $WAIT -ge 300 ]; then
        echo "  ❌ vLLM timed out after 5 min. Check phase4 log."
        cat "${LOG_DIR}/phase4_vllm_ft_${SLURM_JOB_ID}.log" | tail -20
        exit 1
    fi
    echo "  ... waiting ${WAIT}s"
done
echo "  ✅ vLLM serving chainmind-ft on port 8101"
echo "  GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Evaluate fine-tuned model
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 5: Benchmarking Fine-Tuned Model ━━━━━━━━━━━━━━━━━"
echo "  100 tasks × all categories (A, B, C, D)"
echo "  Comparing against: ${BASELINE}"
echo ""

python3 experiments/exp010_qlora_sft/eval_finetuned.py \
    --model-port 8101 \
    --baseline-results "${BASELINE}" \
    --output-dir "${EVAL_OUT}/results" \
    --bench-path chainmind/eval/benchmarks/chainmind_bench.json \
    2>&1 | tee "${LOG_DIR}/phase5_eval_${SLURM_JOB_ID}.log"

# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Final summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━ Phase 6: Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Extract TSR from eval log
FT_TSR=$(grep "OVERALL TSR" "${LOG_DIR}/phase5_eval_${SLURM_JOB_ID}.log" 2>/dev/null | awk '{print $NF}' | head -1)
BASE_TSR=$(grep "Base:" "${LOG_DIR}/phase5_eval_${SLURM_JOB_ID}.log" 2>/dev/null | grep -oP '\d+\.\d+%' | head -1)

echo ""
echo "  Base model TSR:         ${BASE_TSR:-see eval log}"
echo "  Fine-tuned model TSR:   ${FT_TSR:-see eval log}"
echo "  Adapter saved at:       ${MODEL_OUT}/adapter/"
echo "  Merged model at:        ${MODEL_OUT}/merged/"
echo "  Eval results at:        ${EVAL_OUT}/results/"
echo "  Full logs at:           ${LOG_DIR}/phase*.log"
echo ""
echo "  To serve fine-tuned model independently:"
echo "  vllm serve ${FT_MODEL} --port 8101 --served-model-name chainmind-ft"
echo ""
echo "  To run the base model benchmark again (for validation):"
echo "  sbatch run_v5_fixed.sh"

# Kill fine-tuned vLLM
kill ${VLLM_FT_PID} 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   EXP010 PIPELINE COMPLETE                              ║"
echo "║   End: $(date)                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
