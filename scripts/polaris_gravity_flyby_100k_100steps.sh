#!/bin/bash
#SBATCH --job-name=bounce-gravity-flyby-100k-100steps
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-gravity-flyby-100k-100steps-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/gravity_flyby_100k.py --bodies 100000 --steps 100 --truth-steps 0 --record 1 --diag 10 --ckpt 1000 --out results/flyby_100k_100steps.npz
