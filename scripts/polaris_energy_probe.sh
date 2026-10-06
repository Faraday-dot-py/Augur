#!/bin/bash
#SBATCH --job-name=augur-energy-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=augur-energy-probe-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python scripts/energy_probe.py --device cuda --steps 200 --out results/energy_probe.json
