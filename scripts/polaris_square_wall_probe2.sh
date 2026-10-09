#!/bin/bash
#SBATCH --job-name=augur-square-wall-probe2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=augur-square-wall-probe2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-bhtime"
export PYTHONPATH=.
python scripts/square_wall_probe2.py "$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
