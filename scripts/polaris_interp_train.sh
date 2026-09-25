#!/bin/bash
#SBATCH --job-name=interp-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=interp-train-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/interp_train.py --kernels analytic,inv_distance,yukawa30,plw0.5,plw1.5,plw3 --out results/interp_train.json
