#!/bin/bash
#SBATCH --job-name=bounce-gravity-long
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-gravity-long-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p results

python scripts/gravity_long.py --device cuda --no-render --cache results/gravity_long.npz
