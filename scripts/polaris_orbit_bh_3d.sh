#!/bin/bash
#SBATCH --job-name=bounce-orbit-bh-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=bounce-orbit-bh-3d-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/orbit_bh.py --mode blackhole --dim 3 --n 100000 --steps 1000 --substeps 4 --record 8 --diag 5 --ckpt 50 --spill --ratio 0.1 --c 200 --relativistic --sigma 10 --force exact --kernel learned --checkpoint checkpoints/gravity_central_3d.pt --tag bh_3d_100k
