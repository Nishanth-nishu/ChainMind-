#!/usr/bin/env bash
# =============================================================================
# start_model_server.sh — Universal vLLM launcher for ChainMind research
#
# ROOT CAUSE OF PREVIOUS CRASHES (vLLM 0.19.1):
# ─────────────────────────────────────────────
# 1. VLLM_USE_V1=0 has NO effect in vLLM ≥0.8 — v0 engine was removed.
#    vLLM 0.19.1 is v1-only. Setting this env var is misleading and ignored.
#
# 2. --enable-chunked-prefill + --max-model-len 16384 causes the CUDA graph
#    capture worker (EngineCore process) to OOM during the "PIECEWISE" graph
#    set, crashing at ~73% completion. This kills EngineCore before it
#    signals readiness → "Failed core proc(s): {}" with empty dict.
#
# 3. --generation-config vllm does not exist in vLLM 0.19.1 CLI.
#    It causes the APIServer subprocess to die silently on startup.
#
# 4. When background mode starts the server and the 50s wait is too short,
#    the health check fires before the CUDA graph step completes (takes ~90s),
#    so it always reports "failed to start".
#
# THE FIX:
# ─────────
# --enforce-eager          Disables CUDA graph capture entirely.
#                          The PIECEWISE/FULL graph capture is what crashes.
#                          Eager mode is ~10-20% slower per token but 100%
#                          stable. For a research benchmark running 100 tasks
#                          this is the correct tradeoff.
#
# --max-model-len 8192     Halving context window cuts KV cache allocation
#                          by 50%, giving the engine plenty of VRAM headroom.
#                          All 100 D4 tasks fit in ≤4096 tokens.
#
# --gpu-memory-utilization 0.85
#                          Lower than 0.90. Gives the model runner ~3.6 GB
#                          headroom above the base 14.7 GB (Qwen-7B bfloat16)
#                          for activation tensors, hidden states, and output.
#
# NO --enable-chunked-prefill
#                          This flag is the primary trigger for the PIECEWISE
#                          CUDA graph set that crashes. Removed entirely.
#
# NO --generation-config vllm
#                          Not a valid flag in vLLM 0.19.1 CLI. Removed.
#
# Usage:
#   bash scripts/start_model_server.sh qwen2.5-7b          # Foreground
#   bash scripts/start_model_server.sh qwen2.5-7b --bg     # Background
#   bash scripts/start_model_server.sh --list               # Show registry
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate venv
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

# ── Environment ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/scratch/nishanth.r/hf_cache}"
# spawn is required for multiprocessing on Linux with CUDA
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# Do NOT set VLLM_USE_V1 — it has no effect in vLLM 0.19.1 (v1-only)
# Do NOT set VLLM_ENFORCE_EAGER via env — use CLI flag instead

# Load HF_TOKEN from .env if present (needed for gated models like Llama)
if [ -f "$PROJECT_DIR/.env" ]; then
    export $(grep -E '^HF_TOKEN=' "$PROJECT_DIR/.env" | xargs) 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: bash scripts/start_model_server.sh <model_key> [--background|--bg]"
    echo "       bash scripts/start_model_server.sh --list"
    python -m chainmind.eval.model_registry
    exit 0
fi

if [ "$1" = "--list" ]; then
    python -m chainmind.eval.model_registry
    exit 0
fi

MODEL_KEY="$1"
BACKGROUND="${2:-}"

# ---------------------------------------------------------------------------
# Fetch model config from Python registry
# ---------------------------------------------------------------------------
read -r HF_ID SERVED_NAME PORT GPU_MEM DTYPE MAX_LEN MAX_SEQS TRUST_RC DISPLAY <<< "$(python - <<PYEOF
from chainmind.eval.model_registry import get_model
import sys
key = "$MODEL_KEY"
try:
    m = get_model(key)
    trust = "true" if m.trust_remote_code else "false"
    print(m.hf_id, m.served_name, m.port, m.gpu_mem_util,
          m.dtype, m.max_model_len, m.max_num_seqs, trust,
          m.display_name.replace(' ', '_'))
except KeyError as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)"

if [[ "$HF_ID" == ERROR* ]]; then
    echo "$HF_ID"
    exit 1
fi

# Allow env-level override of GPU mem
GPU_MEM="${VLLM_GPU_MEM_UTIL:-$GPU_MEM}"

# ── Safety: cap max_model_len at 8192 to prevent CUDA graph OOM on 24GB VRAM ─
# All ChainMind-Bench tasks fit in ≤4096 tokens. 8192 is a safe upper bound.
SAFE_MAX_LEN=8192
if [ "$MAX_LEN" -gt "$SAFE_MAX_LEN" ]; then
    echo "⚠️  Capping max-model-len from $MAX_LEN → $SAFE_MAX_LEN (prevents CUDA graph OOM)"
    MAX_LEN=$SAFE_MAX_LEN
fi

# ── Safety: cap GPU mem at 0.88 to prevent EngineCore OOM during graph capture ─
GPU_MEM_SAFE=$(python3 -c "print(min(float('$GPU_MEM'), 0.88))")
if [ "$GPU_MEM_SAFE" != "$GPU_MEM" ]; then
    echo "⚠️  Capping GPU mem from $GPU_MEM → $GPU_MEM_SAFE"
    GPU_MEM="$GPU_MEM_SAFE"
fi

echo "============================================================"
echo "  ChainMind Multi-Model Server"
echo "  Model key  : $MODEL_KEY"
echo "  HF ID      : $HF_ID"
echo "  Served as  : $SERVED_NAME"
echo "  Endpoint   : http://0.0.0.0:$PORT/v1"
echo "  GPU mem    : $GPU_MEM (safe-capped)"
echo "  Context    : $MAX_LEN tokens (safe-capped to prevent graph OOM)"
echo "  Mode       : --enforce-eager (CUDA graphs disabled for stability)"
echo "  GPU        : $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader -i 0 2>/dev/null || echo 'unknown')"
echo "============================================================"

# ---------------------------------------------------------------------------
# Kill any stale vLLM process on this port
# ---------------------------------------------------------------------------
if lsof -i:"$PORT" &>/dev/null 2>&1 || ss -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "⚠️  Port $PORT already in use — stopping existing process..."
    pkill -f "vllm.*--port.*$PORT" 2>/dev/null || true
    pkill -f "vllm.*port $PORT"   2>/dev/null || true
    sleep 4
    # Force-kill if still running
    if lsof -i:"$PORT" &>/dev/null 2>&1; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
        sleep 2
    fi
fi

# ---------------------------------------------------------------------------
# Warn for gated models
# ---------------------------------------------------------------------------
if [[ "$HF_ID" == meta-llama/* ]] && [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠️  WARNING: $HF_ID is a gated model."
    echo "   Set HF_TOKEN in .env or visit: https://huggingface.co/$HF_ID"
    echo ""
fi

# ---------------------------------------------------------------------------
# Build vLLM command
#
# Key flags explained:
#   --enforce-eager          Skip CUDA graph capture (prevents EngineCore crash)
#   --max-model-len 8192     Limits KV cache alloc; D4 tasks need ≤4K tokens
#   --gpu-memory-utilization Safe value preventing VRAM OOM during profiling
#   --max-num-seqs 16        Lower concurrency = safer for single-GPU research
#   --disable-log-stats      Suppress per-step stats noise in research logs
# ---------------------------------------------------------------------------
TRUST_FLAG=""
[ "$TRUST_RC" = "true" ] && TRUST_FLAG="--trust-remote-code"

VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$HF_ID"
    --served-model-name "$SERVED_NAME"
    --host "0.0.0.0"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEM"
    --max-model-len "$MAX_LEN"
    --dtype "$DTYPE"
    --enforce-eager
    --distributed-executor-backend uni
    --enable-prefix-caching
    --max-num-seqs 16
    --max-num-batched-tokens 4096
    --disable-log-stats
)
[ -n "$TRUST_FLAG" ] && VLLM_CMD+=("$TRUST_FLAG")

echo "Launch command:"
echo "  ${VLLM_CMD[*]}"
echo ""

# ---------------------------------------------------------------------------
# Background mode
# ---------------------------------------------------------------------------
if [ "$BACKGROUND" = "--background" ] || [ "$BACKGROUND" = "--bg" ] || [ "$BACKGROUND" = "-bg" ]; then
    mkdir -p "$PROJECT_DIR/logs"
    nohup "${VLLM_CMD[@]}" > "$PROJECT_DIR/logs/vllm_${MODEL_KEY}.log" 2>&1 &
    VLLM_PID=$!
    echo "🚀 Started $MODEL_KEY in background (PID=$VLLM_PID)"
    echo "   Log  : $PROJECT_DIR/logs/vllm_${MODEL_KEY}.log"
    echo "   Watch: tail -f $PROJECT_DIR/logs/vllm_${MODEL_KEY}.log"
    echo "   Check: curl http://localhost:$PORT/health"
    echo ""
    echo "   ℹ️  --enforce-eager mode: no CUDA graph capture, startup ~30s"
    exit 0
fi

# ---------------------------------------------------------------------------
# Foreground mode — health-check loop
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "🛑 Shutting down $MODEL_KEY server (PID=$VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

"${VLLM_CMD[@]}" 2>&1 &
VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"
echo "Waiting for server to become ready (enforce-eager → ~30-50s expected)..."
echo ""

MAX_WAIT=360  # 6 min: covers cold model load from NFS (obs: 320s cold, ~30s warm)
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo ""
        echo "✅ $DISPLAY is ready on port $PORT (${WAITED}s)"
        echo ""
        # Warmup request
        echo "Running warmup inference..."
        RESP=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: OK\"}],\"max_tokens\":5,\"temperature\":0}" \
            2>/dev/null || echo "{}")
        if echo "$RESP" | grep -q "OK\|ok\|content"; then
            echo "✅ Warmup OK — model is responding"
        else
            echo "⚠️  Warmup response unexpected: $RESP"
        fi
        echo ""
        echo "  Endpoint : http://localhost:$PORT/v1"
        echo "  Models   : curl http://localhost:$PORT/v1/models"
        echo "  Health   : curl http://localhost:$PORT/health"
        echo ""
        wait "$VLLM_PID"
        exit $?
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    # Show progress every 10s
    if (( WAITED % 10 == 0 )); then
        echo "  ...waiting (${WAITED}s/${MAX_WAIT}s)"
    fi
done

echo ""
echo "❌ Server did not respond within ${MAX_WAIT}s."
echo "   Last 20 lines of output:"
echo "   ---"
kill "$VLLM_PID" 2>/dev/null || true
exit 1
