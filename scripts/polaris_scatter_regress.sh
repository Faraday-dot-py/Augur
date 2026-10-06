#!/bin/bash
#SBATCH --job-name=augur-scatter-regress
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-scatter-regress-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
CKPT="${CKPT:-checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
OUT="${OUT:-results/scatter_baseline.json}"
BASE="${BASE:-}"
EXTRA="${EXTRA:-}"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
python scripts/scatter_regress.py run --ckpt "$CKPT" --out "$OUT" $EXTRA
if [ -n "$BASE" ]; then
  python scripts/scatter_regress.py compare --baseline "$BASE" --candidate "$OUT" || true
fi
