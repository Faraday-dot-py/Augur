#!/bin/bash
#SBATCH --job-name=augur-dE-seeds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-dE-seeds-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${ROOT:-$HOME/augur}"
export PYTHONPATH=.
CKE="${CKE:-checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
CKG="${CKG:-$HOME/edge_ck/ghost_lsq.pt}"
OUT="${OUT:-results/dE_seeds}"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/dE_seeds.py --ckpt-e "$CKE" --ckpt-g "$CKG" --out "$OUT" --n-scenes "${NS:-128}" ${EXTRA:-}
python scripts/dE_seeds_summarize.py "$OUT" "$OUT/summary.txt"
