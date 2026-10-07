#!/bin/bash
#SBATCH --job-name=augur-scatter-s46
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --output=augur-scatter-s46-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur_s46"
cd "$W"
export PYTHONPATH=.
E="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
NAME="${NAME:?}"
FIX="${FIX:-}"
VARIANT="${VARIANT:-ms_kp_pot_v_g128}"
INIT="${INIT:-$E}"
BUDGET="${BUDGET:-1800}"
STRESS="${STRESS:-}"
mkdir -p results checkpoints/scatter_s46
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
CK="$INIT"
if [ -n "$FIX" ]; then
  CK="checkpoints/scatter_s46/$NAME.pt"
  python scripts/train_scatter_ft.py --fix "$FIX" --variant "$VARIANT" --init-ckpt "$INIT" --name "$NAME" --time-budget "$BUDGET" ${TRAIN_EXTRA:-} || echo "TRAIN FAILED"
fi
python scripts/scatter_regress.py run --ckpt "$CK" --variant "$VARIANT" --out "results/${NAME}_regress.json" --battery || echo "REGRESS FAILED"
python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate "results/${NAME}_regress.json" || true
for st in $STRESS; do
  case "$st" in
    ood:*) python scripts/scatter_ood.py --ckpt "$CK" --variant "$VARIANT" --axes "${st#ood:}" --out "results/${NAME}_ood_${st#ood:}.json" || echo "STRESS $st FAILED" ;;
    r46|n46|n46big) python scripts/scatter_scaling.py --ckpt "$CK" --variant "$VARIANT" --phase "$st" --pp cell --out "results/${NAME}_scaling_$st.json" || echo "STRESS $st FAILED" ;;
  esac
done
echo "S46 DONE $NAME"
