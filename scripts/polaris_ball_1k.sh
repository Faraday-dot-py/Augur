#!/bin/bash
#SBATCH --job-name=augur-ball-1k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-ball-1k-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p results

python scripts/ball_1k_rollout.py --device cuda --steps 500
python scripts/ball_1k_rollout.py --device cuda --steps 500 --guard
