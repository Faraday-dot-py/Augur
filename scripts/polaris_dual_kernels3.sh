#!/bin/bash
#SBATCH --job-name=augur-dual-kernels3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-dual-kernels3-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/dual_kernels_run.py --ctrl-mode e2e --ctrl-iters 25 --ctrl-tol 1e-3 --ctrl-rel 0.01 --kernels analytic,yukawa,inv_distance --extra-eval uniform --test-states flyby_t5000 --out results/dual_kernels_e2e.json
