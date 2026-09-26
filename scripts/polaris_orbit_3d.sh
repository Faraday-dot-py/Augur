#!/bin/bash
#SBATCH --job-name=bounce-orbit-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-orbit-3d-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/orbit_3d.py --steps ${STEPS:-130000} --tag ${TAG:-binary3d} --resume
