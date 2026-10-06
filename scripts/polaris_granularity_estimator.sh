#!/bin/bash
#SBATCH --job-name=augur-granularity-est
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-granularity-est-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/granularity_estimator.py --out results/granularity_estimator.json
