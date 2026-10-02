#!/bin/bash
#SBATCH --job-name=bounce-orbit-bh-100k-100steps
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-orbit-bh-100k-100steps-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/orbit_bh.py --mode blackhole --n 100000 --steps 100 --record 1 --diag 10 --ckpt 1000 --c 40 --sigma 10 --force exact --tag bh_100k_100steps
