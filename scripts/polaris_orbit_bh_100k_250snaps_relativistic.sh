#!/bin/bash
#SBATCH --job-name=augur-orbit-bh-100k-250snaps-rel
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-orbit-bh-100k-250snaps-rel-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
python scripts/orbit_bh.py --mode blackhole --n 100000 --steps 31 --substeps 8 --record 1 --diag 1 --ckpt 8 --c 40 --relativistic --sigma 10 --force exact --kernel learned_rel --tag bh_100k_250snaps_relativistic
