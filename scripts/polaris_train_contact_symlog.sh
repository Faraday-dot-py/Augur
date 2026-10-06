#!/bin/bash
#SBATCH --job-name=augur-contact-gen-symlog
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-contact-gen-symlog-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] contact-gen Phase 1 training symlog head (sign+log-scale potential decomposition, same scenes/hparams as v2)"
python -m scripts.train_contact_dynamics --device cuda --seed 4738 --min-bodies 2 --max-bodies 6 \
  --iters 2000 --batch 8 --steps 30 --k-start 4 --k-end 16 --log-every 50 --model symlog \
  --checkpoint checkpoints/contact_dynamics_symlog.pt --out results/contact_dynamics_symlog.json
echo "[$(date -Iseconds)] done"
