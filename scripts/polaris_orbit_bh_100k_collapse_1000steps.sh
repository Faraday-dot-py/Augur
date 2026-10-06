#!/bin/bash
#SBATCH --job-name=augur-orbit-bh-100k-collapse-1000steps
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=augur-orbit-bh-100k-collapse-1000steps-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
python scripts/orbit_bh.py --mode blackhole --n 100000 --steps 1000 --substeps 4 --record 4 --diag 5 --ckpt 50 --spill --ratio 0.1 --c 200 --relativistic --sigma 10 --force exact --kernel learned_rel --tag bh_100k_collapse_1000steps --resume
