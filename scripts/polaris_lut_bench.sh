#!/bin/bash
#SBATCH --job-name=augur-lut-bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-lut-bench-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python scripts/lut_eval.py --out results/lut_eval.json
python scripts/lut_bench.py --out results/lut_bench.json
python scripts/lut_central.py --out results/lut_central.json
