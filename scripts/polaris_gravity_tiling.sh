#!/bin/bash
#SBATCH --job-name=augur-gravity-tiling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-gravity-tiling-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.

pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] training central-force model on 20-100 bodies"
python -m scripts.train_gravity_dynamics --model central --device cuda --seed 4738 --min-bodies 20 --max-bodies 100 --scale-init --train 400 --batch 8 --iters 2000 --log-every 50 --eval-scenes 24 --checkpoint checkpoints/gravity_central_small.pt --out results/gravity_test_central_small.json
echo "[$(date -Iseconds)] scale test"
python -m scripts.gravity_tiling_test --checkpoint checkpoints/gravity_central_small.pt --train-min 20 --train-max 100 --out results/gravity_tiling_test.json
echo "[$(date -Iseconds)] done"
