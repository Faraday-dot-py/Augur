#!/bin/bash
#SBATCH --job-name=augur-scatter-bh-select
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-scatter-bh-select-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
C=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
for ck in "$C" "$C.it55000" "$C.it50000" "$C.it45000"; do
  for seed in 9100 9200 9300 4738; do
    echo "=== $ck seed $seed"
    python scripts/scatter_bh.py --ckpt "$ck" --n 300 --steps 100 --seed $seed --tag "sel_$(basename $ck)_$seed" --device cuda
  done
done
