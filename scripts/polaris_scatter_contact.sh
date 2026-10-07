#!/bin/bash
#SBATCH --job-name=augur-scatter-contact
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-contact-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-contact"
export PYTHONPATH=.
mkdir -p results/contact checkpoints_contact scripts model
cp -rn "$HOME/bounce-scenes/scripts/." scripts/
cp -rn "$HOME/bounce-scenes/model/." model/
ln -sfn "$HOME/bounce-scenes/checkpoints" checkpoints
# STEPS: newline-separated argument lines (full command after python), e.g. "scripts/scatter_contact_scenes.py eval --model E --tag E_zs"
while IFS= read -r step; do
  [ -z "$step" ] && continue
  echo "=== $step"
  python $step || echo "STEP FAILED: $step"
done <<< "$STEPS"
