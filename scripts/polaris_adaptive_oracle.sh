#!/bin/bash
#SBATCH --job-name=augur-adaptive-oracle
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-adaptive-oracle-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python scripts/adaptive_oracle.py --out results/adaptive_oracle.json
