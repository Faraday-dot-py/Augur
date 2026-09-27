#!/bin/bash
#SBATCH --job-name=bounce-lj-v3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=bounce-lj-v3-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.

pip install -q -r requirements.txt
mkdir -p results checkpoints

echo "[$(date -Iseconds)] train"
python -m scripts.train_lj --iters 8000 --k-max 6 --tmax 3.5 --hot-frac 0.15 --hot-tmax 5.0 --scenes 400 --width 128 --checkpoint checkpoints/lj_force_v3.pt --out results/lj_train_v3.json
echo "[$(date -Iseconds)] rollout"
python -m scripts.lj_model_rollout --checkpoint checkpoints/lj_force_v3.pt --tag _v3
echo "[$(date -Iseconds)] eval"
python -m scripts.lj_eval --checkpoint checkpoints/lj_force_v3.pt --tag _v3 --out results/lj_eval_v3.json
echo "[$(date -Iseconds)] done"
