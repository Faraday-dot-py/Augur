#!/bin/bash
#SBATCH --job-name=augur-contact-eval-symlog-capped
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-contact-eval-symlog-capped-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results

echo "[$(date -Iseconds)] contact-gen Phase 1 held-out generalization eval symlog-capped head"
python -m scripts.eval_contact_generalization --checkpoint checkpoints/contact_dynamics_symlog_capped.pt \
  --model symlog_capped --log-scale-cap 4.0 --device cuda --out results/contact_generalization_symlog_capped.json
echo "[$(date -Iseconds)] done"
