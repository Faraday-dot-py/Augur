#!/bin/bash
#SBATCH --job-name=augur-causal-field-2body
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-causal-field-2body-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
python scripts/causal_field_2body.py run --sections ${SECTIONS:-all} --tag "${TAG:-}"
