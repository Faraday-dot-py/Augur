#!/bin/bash
#SBATCH --job-name=augur-scatter-recabl-rec-resume
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-recabl-rec-resume-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cp "$HOME"/recabl_stage/*.py "$HOME/augur-recabl/scripts/"
cd "$HOME/augur-recabl"
export PYTHONPATH=.
VAR=ms_kp_pot_v_g128
SRC=checkpoints/recabl_rec/E
DST=checkpoints/recabl_rec2/E
mkdir -p $DST results/recabl_rec2/E
cp $SRC/E_$VAR.pt.it20000 $DST/E_$VAR.pt
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
# resume of the diverged rec control (3593 went non-finite at it 22720) from its it-20000 snapshot, same recipe; RNG stream restarts at seed+20001
python scripts/train_scatter_field.py --exp E --variant $VAR --train 1500 --steps 50 --batch 16 --k-start 4 --k-end 30 \
  --lr 5e-4 --iters 60769 --eval-steps 100 --log-every 100 --device cuda --seed 4738 \
  --init-ckpt checkpoints/recabl_rec/B/B_$VAR.pt --out-dir results/recabl_rec2/E --ckpt-dir $DST
