#!/bin/bash
#SBATCH --job-name=bounce-dual-kernels2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-dual-kernels2-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/dual_kernels_run.py --kernel-feats 1 --ctrl-iters 25 --out results/dual_kernels_kfeat.json
python scripts/dual_kernels_run.py --kernel-feats 0 --ctrl-iters 25 --kernels analytic,yukawa,inv_distance --out results/dual_kernels_ctrl.json
