#!/bin/bash
#SBATCH --job-name=augur-showcase-bh100k-kpadext
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=augur-showcase-bh100k-kpadext-%j.log

# 100k scatter black hole, learned model with --kpad ext vs exact direct run, seed 4738 (same IC as results/scatter_bh_100000). Runs ~minutes; no checkpoint needed (<1 h).
export PYTHONUNBUFFERED=1
D="$HOME/augur-showcase-kpadext"
cd "$D" && export PYTHONPATH=. && mkdir -p results
python scripts/scatter_bh_large.py --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --n 100000 --steps 100 --kpad ext --tag showcase_bh_100k_kpadext
