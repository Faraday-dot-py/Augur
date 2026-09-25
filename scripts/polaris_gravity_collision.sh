#!/bin/bash
#SBATCH --job-name=bounce-gravity-collision
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-gravity-collision-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/gravity_collision.py --bodies 10000 --steps 3000 --out results/collision_10k.npz
