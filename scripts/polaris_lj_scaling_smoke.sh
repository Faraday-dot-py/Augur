#!/bin/bash
#SBATCH --job-name=bounce-lj-scale-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=bounce-lj-scale-smoke-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.

python -m scripts.lj_scaling --checkpoint checkpoints/lj_force_v2.pt --reps 1 2 4 --frames 20 --warmup 40 --out results/lj_scaling_smoke.json --snaps-out results/lj_scaling_smoke_snaps.npz
echo "[$(date -Iseconds)] done"
