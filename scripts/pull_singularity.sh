#!/usr/bin/env bash
# =============================================================================
# scripts/pull_singularity.sh
# Pull ChainMind Docker image as a Singularity SIF on any SLURM node.
#
# Why Singularity instead of Docker on HPC:
#   - Docker requires root daemon → not allowed on most academic clusters
#   - Singularity runs rootless and is the HPC standard (used at SLURM sites)
#   - `singularity pull` converts Docker Hub images to .sif format automatically
#   - GPU passthrough: `singularity exec --nv` mounts NVIDIA drivers from host
#
# Usage:
#   bash scripts/pull_singularity.sh          # Pull latest
#   bash scripts/pull_singularity.sh v1.1     # Pull specific tag
#   bash scripts/pull_singularity.sh --run    # Pull + run benchmark immediately
#
# After pulling, use these SLURM scripts for each workload:
#   Benchmark:    sbatch scripts/slurm_benchmark_singularity.sh
#   SFT Training: sbatch experiments/exp010_qlora_sft/train_singularity.sh
# =============================================================================
set -euo pipefail

REGISTRY="nishanthr23"
IMAGE="chainmind"
TAG="${1:-latest}"
RUN_AFTER=false
[[ "${1:-}" == "--run" ]] && RUN_AFTER=true && TAG="latest"

DOCKER_URI="docker://docker.io/${REGISTRY}/${IMAGE}:${TAG}"
SIF_DIR="/scratch/nishanth.r/containers"
SIF_FILE="${SIF_DIR}/chainmind_${TAG//:/_}.sif"

echo "=========================================="
echo "  ChainMind Singularity Pull"
echo "  Source: ${DOCKER_URI}"
echo "  Target: ${SIF_FILE}"
echo "=========================================="

mkdir -p "${SIF_DIR}"

# ── Check if singularity is available ─────────────────────────────────────────
if ! command -v singularity &>/dev/null; then
    echo "❌ singularity not found in PATH"
    echo ""
    echo "   Load module: module load singularity"
    echo "   Or load with: module avail | grep -i singularity"
    echo ""
    echo "   If unavailable, contact cluster admins or use:"
    echo "   apptainer pull ${SIF_FILE} ${DOCKER_URI}  (Apptainer = Singularity fork)"
    exit 1
fi
echo "  Singularity: $(singularity --version)"

# ── Pull (or update) ──────────────────────────────────────────────────────────
if [ -f "${SIF_FILE}" ]; then
    EXISTING_SIZE=$(du -sh "${SIF_FILE}" | cut -f1)
    echo ""
    echo "  Existing SIF found: ${SIF_FILE} (${EXISTING_SIZE})"
    echo "  To force re-pull: rm ${SIF_FILE} && bash scripts/pull_singularity.sh"
    echo ""
    echo "  Using existing image."
else
    echo ""
    echo "  Pulling from Docker Hub (first pull: ~10-20 min)..."
    echo "  (Downloads ~6-8GB of layers, then converts to SIF)"
    SINGULARITY_CACHEDIR="/scratch/nishanth.r/.singularity_cache" \
        singularity pull --force "${SIF_FILE}" "${DOCKER_URI}"
    echo "  ✅ SIF created: ${SIF_FILE} ($(du -sh "${SIF_FILE}" | cut -f1))"
fi

# ── Quick validation ──────────────────────────────────────────────────────────
echo ""
echo "  Validating (CPU-only check, no GPU needed)..."
singularity exec "${SIF_FILE}" python3 -c "
import sys
print(f'Python {sys.version.split()[0]}')
import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')
import chainmind; print('ChainMind package: OK')
import rdkit; print('RDKit: OK')
print('✅ Image validation passed')
" && echo "  ✅ Validation OK" || echo "  ⚠️  Validation had warnings (GPU features require --nv flag)"

# ── Print usage instructions ──────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Usage on any SLURM node:"
echo ""
echo "  # Run ChainMind benchmark:"
echo "  singularity exec --nv \\"
echo "      --bind /scratch/nishanth.r:/scratch/nishanth.r \\"
echo "      ${SIF_FILE} \\"
echo "      bash /scratch/nishanth.r/sys_elvle_ai/run_full_selfcontained.sh"
echo ""
echo "  # Run QLoRA fine-tuning:"
echo "  singularity exec --nv \\"
echo "      --bind /scratch/nishanth.r:/scratch/nishanth.r \\"
echo "      ${SIF_FILE} \\"
echo "      bash /scratch/nishanth.r/sys_elvle_ai/experiments/exp010_qlora_sft/train.sh"
echo ""
echo "  # Interactive shell:"
echo "  singularity shell --nv \\"
echo "      --bind /scratch/nishanth.r:/scratch/nishanth.r \\"
echo "      ${SIF_FILE}"
echo "=========================================="

if $RUN_AFTER; then
    echo ""
    echo "── Running benchmark now (--run flag) ──"
    singularity exec --nv \
        --bind /scratch/nishanth.r:/scratch/nishanth.r \
        "${SIF_FILE}" \
        bash /scratch/nishanth.r/sys_elvle_ai/run_full_selfcontained.sh
fi
