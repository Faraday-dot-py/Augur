#!/bin/bash
#SBATCH --job-name=augur-scatter-bh-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-bh-train-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results/scatter_bh checkpoints/scatter_bh
python scripts/train_scatter_field.py --exp E --variant ms_kp_pot_v_g128 --train 1500 --steps 50 --batch 16 --k-start 4 --k-end 30 \
  --lr 5e-4 --time-budget 9000 --eval-steps 100 --log-every 100 --device cuda --seed 4738 \
  --init-ckpt checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt --out-dir results/scatter_bh --ckpt-dir checkpoints/scatter_bh
