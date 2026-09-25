#!/bin/bash
#SBATCH --job-name=bounce-dual-kernels
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-dual-kernels-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/dual_kernels_run.py --out results/dual_kernels.json
