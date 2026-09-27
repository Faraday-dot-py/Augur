#!/bin/bash
#SBATCH --job-name=bounce-lj-scale-smoke2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=bounce-lj-scale-smoke2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.

python -m scripts.lj_scaling --checkpoint checkpoints/lj_force_v2.pt --reps 1 24 --frames 20 --warmup 40 --out results/lj_scaling_smoke2.json --snaps-out results/lj_scaling_smoke2_snaps.npz
echo "[$(date -Iseconds)] done"
