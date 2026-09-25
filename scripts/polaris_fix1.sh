#!/bin/bash
#SBATCH --job-name=bounce-fix1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-fix1-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results
python -m pytest tests/test_token_free.py tests/test_token_model.py -q
python scripts/fix1_probe.py --device cuda
