#!/bin/bash
#SBATCH --job-name=bounce-gravity-1m-far-10k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-gravity-1m-far-10k-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results

python scripts/gravity_1b.py --bodies 1000000 --steps 10000 --record-every 100 --strips 4 --far-grid 1024 --out results/gravity_1m_far_10k.npz
