#!/bin/bash
#SBATCH --job-name=bounce-scatter-scaling-opt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-scatter-scaling-opt-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce_scaling_opt"
export PYTHONPATH=.
CKPT=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
PHASES="${PHASES:-equivcell nsweep worlds}"
mkdir -p results
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
for ph in $PHASES; do
  python scripts/scatter_scaling.py --ckpt "$CKPT" --phase "$ph" --pp cell --out "results/scatter_scaling_${ph}_cell.json" || echo "PHASE $ph FAILED"
done
