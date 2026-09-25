#!/bin/bash
#SBATCH --job-name=bounce-gravity-central
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-gravity-central-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] training central-force model"
python -m scripts.train_gravity_dynamics --model central --device cuda --seed 4738 --min-bodies 100 --max-bodies 1000 --scale-init --train 200 --batch 8 --iters 800 --log-every 20 --eval-scenes 24 --checkpoint checkpoints/gravity_central_v1.pt --out results/gravity_test_central_v1.json
echo "[$(date -Iseconds)] 10k-step rollout"
python scripts/gravity_long.py --model central --checkpoint checkpoints/gravity_central_v1.pt --device cuda --no-render --cache results/gravity_long_central.npz
echo "[$(date -Iseconds)] done"
