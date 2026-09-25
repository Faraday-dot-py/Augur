#!/bin/bash
#SBATCH --job-name=bounce-far-field-accuracy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-far-field-accuracy-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/far_field_accuracy.py --bodies 10000 --steps 2000 --out results/far_field_accuracy.npz
