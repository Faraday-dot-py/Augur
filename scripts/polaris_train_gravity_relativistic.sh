#!/bin/bash
#SBATCH --job-name=bounce-gravity-relativistic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-gravity-relativistic-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] token model on relativistic trajectories (c=1.0)"
python -m scripts.train_gravity_relativistic --model token --device cuda --seed 4738 \
  --min-bodies 3 --max-bodies 8 --c 1.0 --train 2000 --batch 16 --iters 6000 --log-every 200 \
  --eval-scenes 48 --checkpoint checkpoints/gravity_relativistic_token.pt --out results/gravity_relativistic_token.json

echo "[$(date -Iseconds)] central-force baseline (expected structural failure: no v-dependent dv)"
python -m scripts.train_gravity_relativistic --model central --device cuda --seed 4738 \
  --min-bodies 3 --max-bodies 8 --c 1.0 --train 2000 --batch 16 --iters 6000 --log-every 200 \
  --eval-scenes 48 --checkpoint checkpoints/gravity_relativistic_central.pt --out results/gravity_relativistic_central.json

echo "[$(date -Iseconds)] done"
