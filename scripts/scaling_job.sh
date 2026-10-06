#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python "$@"
