#!/bin/bash
#SBATCH --job-name=bounce-scatter-field
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-scatter-field-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results/scatter_field checkpoints/scatter_field
EXP="${EXP:-A}"
ITERS="${ITERS:-2000}"
OUT=results/scatter_field

echo "[$(date -Iseconds)] scatter-field exp $EXP"
python scripts/train_scatter_field.py --exp "$EXP" --variant all --iters "$ITERS" --device cuda

if [ "$EXP" != "C" ]; then
  if [ "$EXP" = "D" ]; then CKSRC=A; else CKSRC=$EXP; fi
  if [ "$EXP" = "A" ] || [ "$EXP" = "D" ]; then NB="--min-bodies 2 --max-bodies 2"; else NB="--min-bodies 10 --max-bodies 100 --scale-init"; fi
  CK=checkpoints/scatter_field/${CKSRC}_baseline_central.pt
  if [ ! -f "$CK" ]; then
    echo "[$(date -Iseconds)] baseline CentralForceDynamics exp $EXP"
    python scripts/train_gravity_dynamics.py --model central $NB --iters 3000 --batch 16 --device cuda \
      --out $OUT/${EXP}_baseline_train.json --checkpoint "$CK"
  fi
  python scripts/train_scatter_field.py --exp "$EXP" --mode baseline --baseline-ckpt "$CK" --device cuda
fi
python scripts/train_scatter_field.py --exp "$EXP" --mode summarize | tee $OUT/${EXP}_summary.md
echo "[$(date -Iseconds)] done"
