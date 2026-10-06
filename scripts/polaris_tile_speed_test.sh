#!/bin/bash
#SBATCH --job-name=augur-tile-speed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-tile-speed-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python scripts/tile_speed_test.py --out results/tile_speed_test.json
