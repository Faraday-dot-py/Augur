#!/bin/bash
#SBATCH --job-name=bounce-gravity-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-gravity-v2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] starting gravity test v2"
python -m scripts.train_gravity_dynamics --seed 4738 --min-bodies 100 --max-bodies 1000 --scale-init --train 200 --batch 2 --iters 800 --log-every 20 --eval-scenes 24 --checkpoint checkpoints/gravity_dynamics_v2.pt --out results/gravity_test_v2.json
echo "[$(date -Iseconds)] done"
