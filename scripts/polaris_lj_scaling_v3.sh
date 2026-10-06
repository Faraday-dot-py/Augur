#!/bin/bash
#SBATCH --job-name=augur-lj-scale-v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=augur-lj-scale-v3-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.

python -m scripts.lj_scaling --checkpoint checkpoints/lj_force_v3.pt --reps 1 2 4 8 16 24 32 --out results/lj_scaling.json --snaps-out results/lj_scaling_big_snaps.npz
echo "[$(date -Iseconds)] done"
