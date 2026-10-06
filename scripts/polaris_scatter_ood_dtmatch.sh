#!/bin/bash
#SBATCH --job-name=augur-scatter-ood2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-scatter-ood2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur-ood/results"
cd "$HOME/augur-ood"
rm -rf scripts model
cp -r "$HOME/augur-opt/scripts" "$HOME/augur-opt/model" .
cp "$HOME/augur-ood-src/scatter_ood.py" scripts/scatter_ood.py
export PYTHONPATH=.
CKPT="${CKPT:-$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
OUT=results/scatter_ood_dtmatch.json
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/scatter_ood.py --ckpt "$CKPT" --out "$OUT" --traj results/scatter_ood_dtmatch_traj.npz --axes mass_dtmatched
