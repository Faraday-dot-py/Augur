#!/bin/bash
#SBATCH --job-name=bounce-tile-box
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=bounce-tile-box-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results
python scripts/tile_box_test.py --out results/tile_box_test.json
