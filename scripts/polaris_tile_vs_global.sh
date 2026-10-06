#!/bin/bash
#SBATCH --job-name=augur-tile-vs-global
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-tile-vs-global-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python scripts/tile_vs_global.py --out results/tile_vs_global.json
