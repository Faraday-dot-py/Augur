#!/bin/bash
#SBATCH --job-name=bounce-lj-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=bounce-lj-v2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.

pip install -q -r requirements.txt
mkdir -p results checkpoints

echo "[$(date -Iseconds)] train"
python -m scripts.train_lj --iters 8000 --k-max 6 --tmax 3.5 --scenes 400 --width 128 --checkpoint checkpoints/lj_force_v2.pt --out results/lj_train_v2.json
echo "[$(date -Iseconds)] rollout"
python -m scripts.lj_model_rollout --checkpoint checkpoints/lj_force_v2.pt --tag _v2
echo "[$(date -Iseconds)] eval"
python -m scripts.lj_eval --checkpoint checkpoints/lj_force_v2.pt --tag _v2 --out results/lj_eval_v2.json
echo "[$(date -Iseconds)] done"
