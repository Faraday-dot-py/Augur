#!/bin/bash
#SBATCH --job-name=augur-contact-gen-symlog-capped
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-contact-gen-symlog-capped-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] contact-gen Phase 1 training symlog-capped head (log_scale soft-capped at +/-4.0, same scenes/hparams as symlog/v2)"
python -m scripts.train_contact_dynamics --device cuda --seed 4738 --min-bodies 2 --max-bodies 6 \
  --iters 2000 --batch 8 --steps 30 --k-start 4 --k-end 16 --log-every 50 --model symlog_capped --log-scale-cap 4.0 \
  --checkpoint checkpoints/contact_dynamics_symlog_capped.pt --out results/contact_dynamics_symlog_capped.json
echo "[$(date -Iseconds)] done"
