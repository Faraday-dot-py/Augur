#!/bin/bash
#SBATCH --job-name=bounce-gravity-clusters
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-gravity-clusters-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results

echo "[$(date -Iseconds)] 1M bodies as 1000 clusters x 1000, timing"
python scripts/gravity_1b.py --bodies 1000000 --clusters 1000 --radius 100 --strips 20 --steps 20 --record-every 5 --out results/gravity_1m_clusters.npz
echo "[$(date -Iseconds)] done"
