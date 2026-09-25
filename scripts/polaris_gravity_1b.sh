#!/bin/bash
#SBATCH --job-name=bounce-gravity-1b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-gravity-1b-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results

echo "[$(date -Iseconds)] 1M validation vs analytic truncated force"
python scripts/gravity_1b.py --bodies 1000000 --steps 100 --record-every 20 --strips 4 --compare-analytic --out results/gravity_1m_check.npz
echo "[$(date -Iseconds)] 1B run"
python scripts/gravity_1b.py --bodies 1000000000 --steps 300 --record-every 5 --strips 100 --max-seconds 10000 --out results/gravity_1b.npz
echo "[$(date -Iseconds)] done"
