#!/bin/bash
#SBATCH --job-name=bounce-flyby-100k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=bounce-flyby-100k-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/gravity_flyby_100k.py --bodies 100000 --dt 0.05 --steps 10000 --truth-steps 10000 --record 40 --diag 250 --ckpt 500 --out results/flyby_100k.npz
