#!/bin/bash
#SBATCH --job-name=augur-scatter-scenes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-scenes-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-scenes"
export PYTHONPATH=.
mkdir -p results/scenes checkpoints_scenes
# STEPS: newline-separated argument lines for scripts/scatter_scenes.py, e.g. "eval --scene bounce --model E --tag zeroshot"
while IFS= read -r step; do
  [ -z "$step" ] && continue
  echo "=== $step"
  python scripts/scatter_scenes.py $step || echo "STEP FAILED: $step"
done <<< "$STEPS"
