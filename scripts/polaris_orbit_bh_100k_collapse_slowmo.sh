#!/bin/bash
#SBATCH --job-name=bounce-orbit-bh-100k-collapse-slowmo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-orbit-bh-100k-collapse-slowmo-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/orbit_bh.py --mode blackhole --n 100000 --steps 14 --substeps 32 --record 1 --diag 1 --ckpt 7 --ratio 0.1 --c 200 --relativistic --sigma 10 --force exact --kernel learned_rel --tag bh_100k_collapse_slowmo
