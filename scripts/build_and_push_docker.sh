#!/usr/bin/env bash
# =============================================================================
# scripts/build_and_push_docker.sh
# Build ChainMind Docker image and push to Docker Hub (nishanthr23)
#
# Usage (from project root):
#   bash scripts/build_and_push_docker.sh [--no-cache] [--tag v1.1]
#
# On SLURM clusters without Docker daemon: use buildah or podman instead.
# The script auto-detects which tool is available.
# =============================================================================
set -euo pipefail

REGISTRY="nishanthr23"
IMAGE="chainmind"
TAG="${1:-latest}"
FULL_TAG="${REGISTRY}/${IMAGE}:${TAG}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NO_CACHE=""
for arg in "$@"; do
    [[ "$arg" == "--no-cache" ]] && NO_CACHE="--no-cache"
    [[ "$arg" == --tag=* ]] && TAG="${arg#--tag=}" && FULL_TAG="${REGISTRY}/${IMAGE}:${TAG}"
done

echo "=========================================="
echo "  ChainMind Docker Build + Push"
echo "  Image:   ${FULL_TAG}"
echo "  Context: ${PROJECT_DIR}"
echo "=========================================="

cd "$PROJECT_DIR"

# ── Detect available container build tool ─────────────────────────────────────
BUILD_TOOL=""
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    BUILD_TOOL="docker"
    echo "Using: docker"
elif command -v buildah &>/dev/null; then
    BUILD_TOOL="buildah"
    echo "Using: buildah (rootless)"
elif command -v podman &>/dev/null; then
    BUILD_TOOL="podman"
    echo "Using: podman (rootless)"
else
    echo "❌ No container build tool found."
    echo "   Install one of: docker, buildah, podman"
    echo ""
    echo "   Alternative (build remotely via GitHub Actions):"
    echo "   Push to GitHub and use .github/workflows/docker.yml"
    exit 1
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Building ${FULL_TAG} ──"
echo "   (First build takes ~30-60 min for Flash Attention compilation)"
echo "   (Subsequent builds use layer cache: ~5 min)"
echo ""

START=$(date +%s)

case "$BUILD_TOOL" in
    docker)
        docker build $NO_CACHE \
            -t "${FULL_TAG}" \
            -t "${REGISTRY}/${IMAGE}:$(date +%Y%m%d)" \
            -f Dockerfile \
            .
        ;;
    buildah)
        buildah build $NO_CACHE \
            -t "${FULL_TAG}" \
            -f Dockerfile \
            .
        ;;
    podman)
        podman build $NO_CACHE \
            -t "${FULL_TAG}" \
            -f Dockerfile \
            .
        ;;
esac

ELAPSED=$(( $(date +%s) - START ))
echo "✅ Build complete in ${ELAPSED}s"

# ── Validate image ────────────────────────────────────────────────────────────
echo ""
echo "── Validating image ──"
case "$BUILD_TOOL" in
    docker)
        docker run --rm "${FULL_TAG}" python3 -c "
import sys
print(f'Python: {sys.version}')
import torch; print(f'PyTorch: {torch.__version__}')
import vllm; print(f'vLLM: {vllm.__version__}')
try:
    import unsloth; print('Unsloth: OK')
except: print('Unsloth: Not available (requires GPU at runtime)')
import rdkit; print('RDKit: OK')
import chainmind; print('ChainMind: OK')
print('✅ Image validation passed')
"
        ;;
    buildah|podman)
        podman run --rm "${FULL_TAG}" python3 -c "import chainmind; print('ChainMind: OK')"
        ;;
esac

# ── Push to Docker Hub ────────────────────────────────────────────────────────
echo ""
echo "── Pushing to Docker Hub ──"
echo "   Target: ${FULL_TAG}"
echo ""
echo "   If not logged in, run first:"
echo "   docker login -u ${REGISTRY}"
echo ""

case "$BUILD_TOOL" in
    docker) docker push "${FULL_TAG}" ;;
    buildah) buildah push "${FULL_TAG}" "docker://docker.io/${FULL_TAG}" ;;
    podman) podman push "${FULL_TAG}" "docker://docker.io/${FULL_TAG}" ;;
esac

# Also push date-tagged version for immutability
if [[ "$TAG" == "latest" ]]; then
    DATE_TAG="${REGISTRY}/${IMAGE}:$(date +%Y%m%d)"
    case "$BUILD_TOOL" in
        docker) docker push "${DATE_TAG}" ;;
        buildah) buildah push "${DATE_TAG}" "docker://docker.io/${DATE_TAG}" ;;
        podman) podman push "${DATE_TAG}" "docker://docker.io/${DATE_TAG}" ;;
    esac
fi

echo ""
echo "=========================================="
echo "  ✅ Push complete!"
echo "  Pull on any node:  singularity pull docker://${FULL_TAG}"
echo "  Run benchmark:     singularity exec --nv chainmind_latest.sif bash run_full_selfcontained.sh"
echo "  Run fine-tuning:   singularity exec --nv chainmind_latest.sif bash experiments/exp010_qlora_sft/train.sh"
echo "=========================================="
