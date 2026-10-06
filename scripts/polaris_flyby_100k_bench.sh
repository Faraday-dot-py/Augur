#!/bin/bash
#SBATCH --job-name=augur-flyby-100k-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=augur-flyby-100k-bench-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
python scripts/gravity_flyby_100k.py --steps 40 --truth-steps 10 --record 10 --diag 10 --ckpt 100000 --out results/flyby_100k_bench.npz
