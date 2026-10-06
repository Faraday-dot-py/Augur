#!/bin/bash
#SBATCH --job-name=augur-gravity-central-far
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-gravity-central-far-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
python scripts/gravity_long.py --model central --checkpoint checkpoints/gravity_central_v1.pt --neighbor-radius 100000 --device cuda --no-render --cache results/gravity_long_central_far.npz
