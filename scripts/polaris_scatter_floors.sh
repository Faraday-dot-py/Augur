#!/bin/bash
#SBATCH --job-name=augur-scatter-floors
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-floors-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "${ROOT:-$HOME/augur}"
export PYTHONPATH=.
OUT="${OUT:-results/chaos_floors.json}"
GROUPS_ARG="${GROUPS_ARG:-}"
python scripts/chaos_floors.py --out "$OUT" ${GROUPS_ARG:+--groups "$GROUPS_ARG"}
