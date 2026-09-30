#!/bin/bash
#SBATCH --job-name=bounce-gravity-central-3d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-gravity-central-3d-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] training 3D central-force model"
python -m scripts.train_gravity_dynamics --model central --dim 3 --device cuda --seed 4738 --min-bodies 100 --max-bodies 1000 --scale-init --train 200 --batch 8 --iters 800 --log-every 20 --eval-scenes 24 --checkpoint checkpoints/gravity_central_3d.pt --out results/gravity_test_central_3d.json
echo "[$(date -Iseconds)] done"
