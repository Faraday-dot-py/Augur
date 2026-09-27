#!/bin/bash
#SBATCH --job-name=bounce-lj-sim
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=bounce-lj-sim-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.

pip install -q -r requirements.txt
mkdir -p results

echo "[$(date -Iseconds)] smoke"
python -m scripts.lj_sim --scenes liquid --nx 12 --ny 14 --steps 400 --out results/lj_smoke
echo "[$(date -Iseconds)] full"
python -m scripts.lj_sim --out results/lj
echo "[$(date -Iseconds)] done"
