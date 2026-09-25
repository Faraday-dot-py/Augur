#!/bin/bash
#SBATCH --job-name=bounce-granularity-est
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-granularity-est-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/granularity_estimator.py --out results/granularity_estimator.json
