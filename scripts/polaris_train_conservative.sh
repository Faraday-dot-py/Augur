#!/bin/bash
#SBATCH --job-name=bounce-cons-pure
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-cons-pure-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p checkpoints results
# checkpoint (model + optimizer + epoch) is saved every epoch by train_conservative.py
python scripts/train_conservative.py --residual 0 --epochs 50 --out checkpoints/cons_pure.pt
