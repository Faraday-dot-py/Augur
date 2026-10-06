#!/bin/bash
#SBATCH --job-name=augur-contact-gen-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-contact-gen-v2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] contact-gen Phase 1 training v2 (post final-review fixes: CENTER offset, sigmoid-gated potential, non-overlapping spawns, gravity*mass)"
python -m scripts.train_contact_dynamics --device cuda --seed 4738 --min-bodies 2 --max-bodies 6 \
  --iters 2000 --batch 8 --steps 30 --k-start 4 --k-end 16 --log-every 50 \
  --checkpoint checkpoints/contact_dynamics_v2.pt --out results/contact_dynamics_v2.json
echo "[$(date -Iseconds)] done"
