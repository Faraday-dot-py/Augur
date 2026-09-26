#!/bin/bash
#SBATCH --job-name=interp-maps2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=interp-maps2-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
date
python scripts/interp_maps.py --out results/interp_maps.json
date
