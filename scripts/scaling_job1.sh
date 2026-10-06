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
python scripts/scaling_opt_bench.py --ns 100000 --tag opt100k
python scripts/scaling_bench.py --states uniform,flyby --ns 10000,30000,100000 --tag a
python scripts/scaling_bench.py --states t10000 --ns 100000 --tag t10k
