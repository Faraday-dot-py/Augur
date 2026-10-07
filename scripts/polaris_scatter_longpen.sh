#!/bin/bash
#SBATCH --job-name=augur-scatter-longpen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-scatter-longpen-%j.log

# Fix 3: warm-start from Exp E, fine-tune with soft dE/dL penalties on long free rollouts (WINDOWS=0 -> control with identical schedule, no penalty), then
# the standing harness + conservation run on the result. Needs TAG; optional WINDOWS, LONGK, LAME, LAML, ITERS.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-fix"
export PYTHONPATH=.
TAG="${TAG:?}"
WINDOWS="${WINDOWS:-5}"
LONGK="${LONGK:-20}"
LAME="${LAME:-0.1}"
LAML="${LAML:-0.1}"
ITERS="${ITERS:-2000}"
LOGEVERY="${LOGEVERY:-50}"
INIT="${INIT:-checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
mkdir -p results/ft_$TAG checkpoints/ft_$TAG
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/train_scatter_field.py --exp E --variant ms_kp_pot_v_g128 --train 1500 --steps 50 --batch 16 --k-start 20 --k-end 20 \
  --lr 1e-4 --iters "$ITERS" --eval-steps 100 --log-every "$LOGEVERY" --device cuda --seed 4738 \
  --long-windows "$WINDOWS" --long-k "$LONGK" --lam-e "$LAME" --lam-l "$LAML" \
  --init-ckpt "$INIT" --out-dir results/ft_$TAG --ckpt-dir checkpoints/ft_$TAG
if [ -z "${NO_EVAL:-}" ]; then
  export CKPT=checkpoints/ft_$TAG/E_ms_kp_pot_v_g128.pt BASE=results/baseline_v2_regress.json
  bash scripts/polaris_scatter_fix.sh
fi
