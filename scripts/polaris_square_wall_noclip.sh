#!/bin/bash
#SBATCH --job-name=augur-square-wall-noclip2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=augur-square-wall-noclip2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-bhtime"
export PYTHONPATH=.
mkdir -p results
python scripts/scatter_bh_1b.py --ckpt "$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt" --n 1000000000 --steps 30 --hist --snaps --ic-resample --mom-scale 16384 --tag scatter_bh_1b_noclip
echo DONE
