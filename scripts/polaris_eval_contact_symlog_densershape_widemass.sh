#!/bin/bash
#SBATCH --job-name=bounce-contact-eval-symlog-densershape-widemass
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-contact-eval-symlog-densershape-widemass-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results

echo "[$(date -Iseconds)] contact-gen Phase 1 held-out generalization eval symlog head (denser-obstacle-shape + widened mass-range training variant)"
python -m scripts.eval_contact_generalization --checkpoint checkpoints/contact_dynamics_symlog_densershape_widemass.pt \
  --model symlog --device cuda --out results/contact_generalization_symlog_densershape_widemass.json
echo "[$(date -Iseconds)] done"

echo "[$(date -Iseconds)] contact force probes: boundary case (1 vs 10) and stress case (1 vs 30)"
python -m scripts.contact_force_probe --checkpoint checkpoints/contact_dynamics_symlog_densershape_widemass.pt \
  --model symlog --mass1 1.0 --mass2 10.0
python -m scripts.contact_force_probe --checkpoint checkpoints/contact_dynamics_symlog_densershape_widemass.pt \
  --model symlog --mass1 1.0 --mass2 30.0
echo "[$(date -Iseconds)] done"
