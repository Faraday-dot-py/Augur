#!/bin/bash
#SBATCH --job-name=bounce-scatter-scaling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-scatter-scaling-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce_scaling"
export PYTHONPATH=.
CKPT="${CKPT:-checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
PHASES="${PHASES:-equiv stock comp nsweep worlds}"
mkdir -p results
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
for ph in $PHASES; do
  python scripts/scatter_scaling.py --ckpt "$CKPT" --phase "$ph" --out "results/scatter_scaling_${ph}${TAGSUF:-}.json" || echo "PHASE $ph FAILED"
done
