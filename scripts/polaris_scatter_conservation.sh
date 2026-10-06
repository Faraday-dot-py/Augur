#!/bin/bash
#SBATCH --job-name=augur-scatter-cons
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-scatter-cons-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-cons"
export PYTHONPATH=.
CKPT="${CKPT:-$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
STEPS="${STEPS:-10000}"
REGS="${REGS:-nr2,nr10,nr100,nr300,bh10,bh100,bh300,bh1000}"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/scatter_conservation.py probe --ckpt "$CKPT"
python scripts/scatter_conservation.py run --ckpt "$CKPT" --steps "$STEPS" --regimes "$REGS"
