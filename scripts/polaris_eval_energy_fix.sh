#!/bin/bash
#SBATCH --job-name=augur-eval-energy-fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-eval-energy-fix-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
MODELS="soupb=soupb:checkpoints/token_model_soup_b.pt ${EXTRA_MODELS:-}"
python scripts/eval_energy_fix.py --models $MODELS --seeds 9000 12000 --out results/eval_energy_fix.json
python scripts/ball_1k_fix.py --models $MODELS --steps 500 --out results/ball_1k_fix.npz
