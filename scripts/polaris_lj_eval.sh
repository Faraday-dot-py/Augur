#!/bin/bash
#SBATCH --job-name=bounce-lj-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=bounce-lj-eval-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.

pip install -q -r requirements.txt
mkdir -p results checkpoints

echo "[$(date -Iseconds)] rollout"
python -m scripts.lj_model_rollout
echo "[$(date -Iseconds)] eval"
python -m scripts.lj_eval
echo "[$(date -Iseconds)] done"
