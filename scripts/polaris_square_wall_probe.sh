#!/bin/bash
#SBATCH --job-name=augur-square-wall-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=augur-square-wall-probe-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-bhtime"
export PYTHONPATH=.
python scripts/square_wall_probe.py "$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
