#!/bin/bash
#SBATCH --job-name=augur-scatter-joint
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=augur-scatter-joint-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_joint"
export PYTHONPATH=.
V=ms_kp_pot_v_g128_cu_eps_mnorm
NAME="${NAME:?}"
FIX="${FIX:-all}"
BUDGET="${BUDGET:-12000}"
LR="${LR:-5e-4}"
STEPS="${STEPS:-100}"
KEND="${KEND:-40}"
LONGW="${LONGW:-2}"
STRESS="${STRESS-ood:all r46 n46 n46big}"
BASE=results/baseline_v2_regress.json
mkdir -p results checkpoints/scatter_joint
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
ck="checkpoints/scatter_joint/$NAME.pt"
if [ -n "${CHECKS:-}" ]; then
  python -m pytest -q -x tests 2>&1 | tail -15
fi
python scripts/train_scatter_ft.py --fix "$FIX" --variant "$V" --name "$NAME" --time-budget "$BUDGET" --lr "$LR" --steps "$STEPS" --k-end "$KEND" \
  --loss-norm --long-windows "$LONGW" --out-dir results/scatter_joint --ckpt-dir checkpoints/scatter_joint ${TRAIN_EXTRA:-} || echo "TRAIN FAILED"
python scripts/scatter_regress.py run --ckpt "$ck" --variant "$V" --out "results/${NAME}_regress.json" --battery || echo "REGRESS FAILED"
python scripts/scatter_regress.py compare --baseline "$BASE" --candidate "results/${NAME}_regress.json" || true
for st in $STRESS; do
  case "$st" in
    ood:all) python scripts/scatter_ood.py --ckpt "$ck" --variant "$V" --out "results/${NAME}_ood_all.json" --opt kcache,fastio,cl || echo "STRESS $st FAILED" ;;
    *) python scripts/scatter_scaling.py --ckpt "$ck" --variant "$V" --phase "$st" --pp cell --out "results/${NAME}_scaling_$st.json" || echo "STRESS $st FAILED" ;;
  esac
done
echo "JOINT DONE $NAME"
