#!/bin/bash
#SBATCH --job-name=augur-gravity-cic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-gravity-cic-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results

python scripts/gravity_1b.py --bodies 1000000 --steps 10000 --record-every 100 --strips 4 --far-grid 1024 --out results/gravity_1m_far_cic_10k.npz
python scripts/gravity_collision.py --bodies 10000 --steps 3000 --out results/collision_cic_10k.npz
python scripts/gravity_collision.py --bodies 10000 --steps 4000 --separation 400 --impact 120 --approach 5 --spin 4 --out results/flyby_cic_10k.npz
