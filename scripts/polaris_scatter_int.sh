#!/bin/bash
#SBATCH --job-name=augur-scatter-int
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=augur-scatter-int-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_s46"
export PYTHONPATH=.
E="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
V=ms_kp_pot_v_g128_cu_eps_mnorm
NAME="${NAME:?}"
BUDGET="${BUDGET:-9000}"
REPLAY="${REPLAY:-0.4}"
LR="${LR:-2e-4}"
STRESS="${STRESS:-ood:all r46 n46 n46big}"
BASE=results/baseline_v2_regress.json
mkdir -p results checkpoints/scatter_s46
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
ck="checkpoints/scatter_s46/$NAME.pt"
if [ -n "${CHECKS:-}" ]; then
  python -m pytest -q -x tests 2>&1 | tail -15
  python scripts/scatter_s46_smoke.py --ckpt "$E" 2>&1 | tail -30
fi
python scripts/train_scatter_ft.py --fix all --variant "$V" --init-ckpt "$E" --name "$NAME" --time-budget "$BUDGET" --replay "$REPLAY" --lr "$LR" ${TRAIN_EXTRA:-} || echo "TRAIN FAILED"
python scripts/scatter_regress.py run --ckpt "$ck" --variant "$V" --out "results/${NAME}_regress.json" --battery || echo "REGRESS FAILED"
python scripts/scatter_regress.py compare --baseline "$BASE" --candidate "results/${NAME}_regress.json" || true
for st in $STRESS; do
  case "$st" in
    ood:all) python scripts/scatter_ood.py --ckpt "$ck" --variant "$V" --out "results/${NAME}_ood_all.json" --opt kcache,fastio,cl || echo "STRESS $st FAILED" ;;
    ood:*) python scripts/scatter_ood.py --ckpt "$ck" --variant "$V" --axes "${st#ood:}" --out "results/${NAME}_ood_${st#ood:}.json" --opt kcache,fastio,cl || echo "STRESS $st FAILED" ;;
    *) python scripts/scatter_scaling.py --ckpt "$ck" --variant "$V" --phase "$st" --pp cell --out "results/${NAME}_scaling_$st.json" || echo "STRESS $st FAILED" ;;
  esac
done
echo "INT DONE $NAME"
