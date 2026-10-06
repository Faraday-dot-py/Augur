#!/bin/bash
#SBATCH --job-name=bounce-scatter-ood
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-scatter-ood-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/bounce-ood/results"
cd "$HOME/bounce-ood"
rm -rf scripts model
cp -r "$HOME/bounce-opt/scripts" "$HOME/bounce-opt/model" .
cp "$HOME/bounce-ood-src/scatter_ood.py" scripts/scatter_ood.py
export PYTHONPATH=.
CKPT="${CKPT:-$HOME/bounce/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
OUT="${OUT:-results/scatter_ood.json}"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/scatter_ood.py --ckpt "$CKPT" --out "$OUT" --traj "${TRAJ:-results/scatter_ood_traj.npz}" ${EXTRA:-}
