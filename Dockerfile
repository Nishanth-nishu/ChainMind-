# =============================================================================
# ChainMind Docker Image
# Portable environment for benchmarking + fine-tuning on any SLURM node
#
# Build:  docker build -t nishanthr23/chainmind:latest .
# Push:   docker push nishanthr23/chainmind:latest
# Run:    singularity exec --nv chainmind_latest.sif bash run_full_selfcontained.sh
#
# Design decisions:
#   - CUDA 12.1 + cuDNN 8 (devel) for Flash Attention 2 compilation
#   - RTX 3090: Ampere arch (sm_86), needs CUDA >= 11.1, optimal at 12.x
#   - Model weights NOT included (too large). Mount /scratch at runtime.
#   - Two-stage build: deps in builder → lean runtime image
#   - Ubuntu 22.04 LTS for best CUDA + Python 3.10 support
# =============================================================================

# ── Stage 1: Builder (install + compile all deps) ────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git git-lfs curl wget \
    build-essential cmake ninja-build \
    librdkit-dev \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 → python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create isolated venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip + wheel
RUN pip install --upgrade pip setuptools wheel

# ── PyTorch 2.3.0 with CUDA 12.1 ────────────────────────────────────────────
# Must install before flash-attn (depends on torch version)
RUN pip install torch==2.3.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

# ── Flash Attention 2 (compile from source for RTX 3090 / sm_86) ─────────────
# Required by: vLLM, Unsloth, transformers attention_implementation="flash_attention_2"
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN pip install flash-attn --no-build-isolation --no-cache-dir

# ── vLLM: LLM inference server ───────────────────────────────────────────────
RUN pip install vllm==0.4.3 --no-cache-dir

# ── HuggingFace training stack ───────────────────────────────────────────────
RUN pip install \
    transformers==4.43.0 \
    datasets==2.20.0 \
    accelerate==0.31.0 \
    peft==0.11.1 \
    "trl>=0.12.0" \
    bitsandbytes==0.43.1 \
    sentencepiece \
    tiktoken \
    --no-cache-dir

# ── Unsloth (2x faster QLoRA) ────────────────────────────────────────────────
# Install without build isolation (uses torch already installed above)
RUN pip install \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    xformers \
    --no-deps --no-cache-dir

# ── ChainMind runtime dependencies ───────────────────────────────────────────
RUN pip install \
    rdkit-pypi \
    pubchempy \
    pydantic==2.7.4 \
    pydantic-settings==2.3.4 \
    openai==1.35.7 \
    httpx \
    tenacity \
    duckduckgo-search \
    chromadb \
    sentence-transformers \
    numpy scipy pandas \
    matplotlib seaborn \
    loguru \
    rich \
    --no-cache-dir

# ── Stage 2: Runtime (lean final image) ──────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip \
    git curl wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Copy the full venv from builder (all compiled packages)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy cuDNN shared libs from devel image (needed by flash-attn + bitsandbytes at runtime)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/cuda-12.1/lib64/libcusparse* /usr/local/cuda-12.1/lib64/
COPY --from=builder /usr/local/cuda-12.1/lib64/libcublas* /usr/local/cuda-12.1/lib64/

# ── ChainMind source code ─────────────────────────────────────────────────────
WORKDIR /workspace/chainmind
COPY . /workspace/chainmind/

# Set environment variables
ENV PYTHONPATH="/workspace/chainmind"
ENV HF_HOME="/scratch/nishanth.r/hf_cache"
ENV ENVIRONMENT="development"
ENV CUDA_VISIBLE_DEVICES=0

# Validation: quick import check
RUN python3 -c "import torch; import vllm; import unsloth; import rdkit; print('✅ All imports OK')" \
    || echo "⚠️ Some imports failed — check CUDA availability at runtime (GPU required)"

ENTRYPOINT ["/bin/bash"]
CMD []

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="ChainMind"
LABEL org.opencontainers.image.description="Agentic drug discovery benchmarking + QLoRA fine-tuning"
LABEL org.opencontainers.image.source="https://github.com/Nishanth-nishu/ChainMind-"
LABEL cuda.version="12.1"
LABEL pytorch.version="2.3.0"
