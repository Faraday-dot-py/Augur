#!/bin/bash
#SBATCH --job-name=augur-flyby-100k-dt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=augur-flyby-100k-dt-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
python scripts/gravity_flyby_100k.py --dt 0.05 --steps 200 --truth-steps 0 --diag 50 --ckpt 100000 --out results/flyby_100k_dt05.npz
python scripts/gravity_flyby_100k.py --dt 0.02 --steps 500 --truth-steps 0 --diag 125 --ckpt 100000 --out results/flyby_100k_dt02.npz
python scripts/gravity_flyby_100k.py --dt 0.1 --steps 100 --truth-steps 0 --diag 25 --ckpt 100000 --out results/flyby_100k_dt10.npz
