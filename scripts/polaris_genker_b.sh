#!/bin/bash
#SBATCH --job-name=genker-b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=genker-b-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints


python scripts/genker_static.py --kernels inv_distance,yukawa30,learned --out results/genker_static_b.json
