#!/bin/bash
#SBATCH --job-name=augur-contact-gen-symlog-densershape
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-contact-gen-symlog-densershape-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] contact-gen Phase 1 training symlog head (uncapped) + denser/wider obstacle-shape sampling (obstacle_prob 0.8, circle radius 2-9, spacing 0.6-3.0)"
python -m scripts.train_contact_dynamics --device cuda --seed 4738 --min-bodies 2 --max-bodies 6 \
  --iters 2000 --batch 8 --steps 30 --k-start 4 --k-end 16 --log-every 50 --model symlog \
  --obstacle-prob 0.8 --circle-radius-min 2.0 --circle-radius-max 9.0 --spacing-min 0.6 --spacing-max 3.0 \
  --checkpoint checkpoints/contact_dynamics_symlog_densershape.pt --out results/contact_dynamics_symlog_densershape.json
echo "[$(date -Iseconds)] done"
