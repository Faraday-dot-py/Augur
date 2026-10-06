#!/bin/bash
#SBATCH --job-name=augur-granularity-abl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-granularity-abl-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/granularity_estimator.py --epochs 5 --out results/granularity_estimator_ep5.json
python scripts/granularity_estimator.py --train-states flyby_t1000,flyby_t5000,flyby_t10000 --test-states uniform,flyby_t0 --out results/granularity_estimator_rev.json
