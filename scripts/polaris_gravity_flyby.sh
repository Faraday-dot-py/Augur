#!/bin/bash
#SBATCH --job-name=bounce-gravity-flyby
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-gravity-flyby-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
python scripts/gravity_collision.py --bodies 10000 --steps 4000 --separation 400 --impact 120 --approach 5 --spin 4 --out results/flyby_10k.npz
