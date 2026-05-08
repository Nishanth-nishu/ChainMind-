#!/usr/bin/env bash
# =============================================================================
# EXP010: QLoRA SFT SLURM Script
# Fine-tunes Qwen2.5-7B-Instruct on ChainMind domain data using Unsloth.
#
# GPU Usage: Full 24GB VRAM during training. vLLM must be OFF.
# Wall time:  ~5-8h total (dataset build + training + eval)
#
# Submit: sbatch experiments/exp010_qlora_sft/train.sh
# Monitor: tail -f logs/exp010_qlora_sft_<JOB_ID>.log
# =============================================================================
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J CM_SFT_v1
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode118
#SBATCH --mem-per-cpu=3G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/nishanth.r/sys_elvle_ai/logs/exp010_qlora_sft_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

set -euo pipefail

echo "=========================================="
echo "  EXP010: QLoRA SFT — ChainMind-FT-v1"
echo "  SLURM_JOB_ID  = $SLURM_JOB_ID"
echo "  NODE          = $SLURM_NODELIST"
echo "  START         = $(date)"
echo "=========================================="

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR="/scratch/nishanth.r/sys_elvle_ai"
VENV="${PROJECT_DIR}/.venv"
LOG_DIR="${PROJECT_DIR}/logs"
MODEL_OUT="${PROJECT_DIR}/models/chainmind-ft-v1"

# ── Environment ────────────────────────────────────────────────────────────
export HF_HOME="/scratch/nishanth.r/hf_cache"
export PIP_CACHE_DIR="/scratch/nishanth.r/.home_cache_migration/pip"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export ENVIRONMENT="development"
export CUDA_VISIBLE_DEVICES=0

# Flash Attention 2 — required for Unsloth on Ampere GPUs
export FLASH_ATTENTION_FORCE_BUILD=1

mkdir -p "${LOG_DIR}" "${MODEL_OUT}"
cd "$PROJECT_DIR"

# ── Kill any running vLLM (CRITICAL: must free GPU before training) ─────────
echo ""
echo "── Step 0: Freeing GPU from vLLM ──"
if lsof -ti:8100 > /dev/null 2>&1; then
    echo "  Killing vLLM on port 8100..."
    kill "$(lsof -ti:8100)" 2>/dev/null || true
    sleep 5
fi
echo "  GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)MB available"

# ── Activate venv ──────────────────────────────────────────────────────────
source "${VENV}/bin/activate"
echo "Python: $(which python) | $(python --version)"

# ── Install training dependencies if missing ───────────────────────────────
echo ""
echo "── Step 1: Checking training dependencies ──"

install_if_missing() {
    local pkg=$1; local install_cmd=${2:-"pip install $pkg"}
    if ! python3 -c "import ${pkg//-/_}" 2>/dev/null; then
        echo "  Installing $pkg..."
        eval "$install_cmd"
    else
        echo "  $pkg: ✅"
    fi
}

# bitsandbytes: GPU quantization (must come before unsloth)
install_if_missing "bitsandbytes" "pip install bitsandbytes --upgrade"

# unsloth: 2x faster QLoRA (install CUDA-specific build)
if ! python3 -c "import unsloth" 2>/dev/null; then
    echo "  Installing unsloth (CUDA 12.1 build)..."
    pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git" \
        --no-deps 2>&1 | tail -3
    pip install xformers --no-deps 2>&1 | tail -2
fi
echo "  unsloth: ✅"

# trl: SFTTrainer + DPOTrainer + GRPOTrainer
install_if_missing "trl" "pip install 'trl>=0.12.0'"

# accelerate + peft (HuggingFace training stack)
install_if_missing "accelerate" "pip install accelerate"
install_if_missing "peft" "pip install peft"
install_if_missing "datasets" "pip install datasets"

echo ""
echo "── Step 2: GPU Status ──"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader

# ── EXP009: Build SFT dataset ─────────────────────────────────────────────
echo ""
echo "── Step 3: Building SFT Dataset (EXP009) ──"
SFT_DATA="${PROJECT_DIR}/data/sft_dataset.jsonl"

if [ -f "${SFT_DATA}" ]; then
    N_EXAMPLES=$(wc -l < "${SFT_DATA}")
    echo "  SFT dataset already exists: ${N_EXAMPLES} examples. Skipping rebuild."
    echo "  (Delete ${SFT_DATA} to force rebuild)"
else
    echo "  Building from benchmark traces + Mol-Instructions..."
    python3 scripts/build_sft_dataset.py \
        --max-mol-instructions 5000 \
        2>&1 | tee "${LOG_DIR}/exp009_build_data_${SLURM_JOB_ID}.log"
    N_EXAMPLES=$(wc -l < "${SFT_DATA}")
    echo "  ✅ Dataset built: ${N_EXAMPLES} examples"
fi

# Validate minimum dataset size
if [ "${N_EXAMPLES:-0}" -lt 100 ]; then
    echo "❌ Dataset too small (<100 examples). Cannot train. Check EXP009 logs."
    exit 1
fi

# ── EXP010: QLoRA Training ─────────────────────────────────────────────────
echo ""
echo "── Step 4: QLoRA SFT Training (EXP010) ──"
echo "  Output:       ${MODEL_OUT}"
echo "  Data:         ${SFT_DATA} (${N_EXAMPLES} examples)"
echo "  Config:       r=64, alpha=128, max_seq_len=2048, lr=2e-4, epochs=2"
echo "  Expected:     3-5h training, ~12-14GB VRAM"
echo ""

python3 experiments/exp010_qlora_sft/train.py \
    --output-dir "${MODEL_OUT}" \
    --lora-r 64 \
    --lora-alpha 128 \
    --max-seq-length 2048 \
    --learning-rate 2e-4 \
    --epochs 2 \
    2>&1 | tee "${LOG_DIR}/exp010_train_${SLURM_JOB_ID}.log"

echo ""
echo "── Step 5: Post-Training Validation ──"
if [ -f "${MODEL_OUT}/adapter/adapter_model.safetensors" ]; then
    ADAPTER_SIZE=$(du -sh "${MODEL_OUT}/adapter/" | cut -f1)
    echo "  ✅ LoRA adapter saved: ${MODEL_OUT}/adapter/ (${ADAPTER_SIZE})"
else
    echo "  ❌ Adapter not found — training may have failed. Check logs."
    exit 1
fi

if [ -d "${MODEL_OUT}/merged" ]; then
    MERGED_SIZE=$(du -sh "${MODEL_OUT}/merged/" | cut -f1)
    echo "  ✅ Merged model saved: ${MODEL_OUT}/merged/ (${MERGED_SIZE})"
fi

# ── Quick model sanity test ───────────────────────────────────────────────
echo ""
echo "── Step 6: Model Sanity Test ──"
python3 - <<'PYEOF'
import os, sys
os.environ["HF_HOME"] = "/scratch/nishanth.r/hf_cache"

from unsloth import FastLanguageModel

model_path = "models/chainmind-ft-v1/adapter"
print(f"Loading from {model_path}...")

# Quick test: load adapter and run one inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=512,
    load_in_4bit=True,
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, model_path)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "Does Aspirin (SMILES: CC(=O)OC1=CC=CC=C1C(=O)O) pass Lipinski Rule of 5?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True)
response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Test response: {response[:200]}")
print("✅ Model sanity test passed!")
PYEOF

echo ""
echo "=========================================="
echo "  EXP010 COMPLETE"
echo "  Models at: ${MODEL_OUT}/"
echo "  To serve:  vllm serve ${MODEL_OUT}/merged --port 8100 --served-model-name chainmind-ft-v1"
echo "  END = $(date)"
echo "=========================================="
