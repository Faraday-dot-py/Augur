#!/bin/bash
#SBATCH --job-name=bounce-sf-iter
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-sf-iter-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
EXP="${EXP:-A}"
ITERS="${ITERS:-4000}"
VARIANTS="${VARIANTS:?}"
echo "[$(date -Iseconds)] sf-iter exp $EXP variants $VARIANTS"
python scripts/train_scatter_field.py --exp "$EXP" --variant "$VARIANTS" --iters "$ITERS" --device cuda
python scripts/train_scatter_field.py --exp "$EXP" --mode summarize | tee results/scatter_field/${EXP}_summary_iter.md
echo "[$(date -Iseconds)] done"
