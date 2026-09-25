#!/bin/bash
#SBATCH --job-name=bounce-far-field-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=bounce-far-field-check-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/far_field_check.py --bodies 20000
