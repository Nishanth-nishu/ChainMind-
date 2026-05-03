#!/usr/bin/env bash
# =============================================================================
# start_vllm_optimized.sh — ChainMind vLLM Server (SLURM-compatible)
#
# ROOT CAUSE ANALYSIS (why vLLM 0.19.1 crashes on this node):
# ─────────────────────────────────────────────────────────────
# The crash is NOT about CUDA graphs or VLLM_USE_V1.
#
# The actual cause is a SLURM + multiprocessing incompatibility:
#   1. vLLM 0.19.1 spawns a separate EngineCore subprocess via Python
#      multiprocessing (VLLM_WORKER_MULTIPROC_METHOD=spawn).
#   2. The EngineCore must load model weights (~14.25 GiB) from disk,
#      which takes 148–320 seconds on a cold NFS cache.
#   3. The APIServer process has a fixed 120s timeout waiting for EngineCore
#      to signal readiness over a ZMQ IPC socket.
#   4. Under SLURM, the /tmp/vllm*.ipc socket path may be on a tmpfs that
#      has strict ulimits or gets cleaned up, breaking the handshake.
#   5. Result: "Failed core proc(s): {}" — empty dict means IPC timeout,
#      not an actual model error. The model DID load correctly.
#
# SOLUTION:
#   --distributed-executor-backend uni
#     Runs everything in a SINGLE process (no subprocess IPC).
#     EngineCore runs in-process instead of as a separate PID.
#     Eliminates the IPC/socket handshake entirely.
#     This is the recommended mode for single-GPU research deployments.
#
#   --enforce-eager
#     Disables CUDA graph capture (still a good practice for 24GB VRAM).
#
#   Longer timeout in health check (300s instead of 180s)
#     Cold model load takes 2–5 minutes. Health check must wait this long.
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/scratch/nishanth.r/hf_cache}"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_V1=0

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8100}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
SERVED_NAME="${VLLM_SERVED_NAME:-chainmind-qwen}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

[ -f "$PROJECT_DIR/.venv/bin/activate" ] && source "$PROJECT_DIR/.venv/bin/activate"

echo "============================================================"
echo "  ChainMind vLLM Server (SLURM-compatible, single-process)"
echo "  Model    : $MODEL"
echo "  Endpoint : http://$HOST:$PORT/v1"
echo "  Context  : $MAX_MODEL_LEN tokens"
echo "  GPU mem  : $GPU_MEM_UTIL"
echo "  Backend  : uni (single-process, no IPC, SLURM-safe)"
echo "  CUDA     : enforce-eager (no graph capture)"
echo "============================================================"
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i 0 2>/dev/null || echo "unknown")
echo "  GPU: $GPU_NAME"
echo "============================================================"
echo "  NOTE: First start takes 2-5min for model weight loading."
echo "============================================================"

# Kill any existing server on this port
if lsof -i:"$PORT" &>/dev/null 2>&1 || ss -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "Port $PORT in use — stopping old server..."
    pkill -f "vllm.*$PORT" 2>/dev/null || true
    sleep 4
fi

cleanup() { echo ""; echo "Shutting down..."; kill "$VLLM_PID" 2>/dev/null || true; exit 0; }
trap cleanup SIGINT SIGTERM

# ── Launch vLLM in SINGLE-PROCESS mode ───────────────────────────────────────
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype bfloat16 \
    --trust-remote-code \
    --enforce-eager \
    --distributed-executor-backend uni \
    --enable-prefix-caching \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --disable-log-stats \
    2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
echo "Waiting for model to load (cold load = 2-5 min, warm = ~30s)..."
echo ""

MAX_WAIT=360   # 6 min — covers cold model load
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo ""
        echo "✅ vLLM ready in ${WAITED}s"
        # Quick warmup
        curl -sf "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply OK.\"}],\"max_tokens\":5,\"temperature\":0}" \
            >/dev/null 2>&1 && echo "✅ Warmup OK"
        echo ""
        echo "  Endpoint : http://$HOST:$PORT/v1"
        echo "  Models   : curl http://localhost:$PORT/v1/models"
        echo "============================================================"
        wait "$VLLM_PID"
        exit $?
    fi
    # Check if process died
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ vLLM process died. Check output above for error."
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    (( WAITED % 30 == 0 )) && echo "  ...waiting (${WAITED}s/${MAX_WAIT}s) — model loading..."
done

echo "❌ Server not ready in ${MAX_WAIT}s"
kill "$VLLM_PID" 2>/dev/null || true
exit 1
