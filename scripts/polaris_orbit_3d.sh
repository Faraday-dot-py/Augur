#!/bin/bash
#SBATCH --job-name=augur-orbit-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-orbit-3d-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
python scripts/orbit_3d.py --steps ${STEPS:-130000} --tag ${TAG:-binary3d} --resume
