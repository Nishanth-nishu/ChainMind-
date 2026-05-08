#!/bin/bash
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -J EGNN_GEOMETRY
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gnode118
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=jepa_geometry_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nishanth0962333@gmail.com

echo "=========================================="
echo "SLURM_JOB_ID    = $SLURM_JOB_ID"
echo "SLURM_NODELIST = $SLURM_NODELIST"
echo "SLURM_JOB_GPUS = $SLURM_JOB_GPUS"
echo "START TIME     = $(date)"
echo "=========================================="

# --------------------------------------------------
# Move to project directory (SLURM starts in $HOME)
# --------------------------------------------------
cd /scratch/nishanth.r/new_egnn/egnn/ || exit 1
echo "Working directory: $(pwd)"

# --------------------------------------------------
# Proper Conda initialization (FIXED)
# --------------------------------------------------

source venv/bin/activate
echo "Activated Conda environment:"
which python
python --version

# Sanity check (VERY IMPORTANT)
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# --------------------------------------------------
# Run training
# --------------------------------------------------
echo "Starting EGNN geometry training..."
python train_geo_com_op.py 

echo "=========================================="
echo "JOB COMPLETED"
echo "END TIME = $(date)"
echo "=========================================="
