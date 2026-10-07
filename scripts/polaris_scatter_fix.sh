#!/bin/bash
#SBATCH --job-name=augur-scatter-fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-scatter-fix-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-fix"
export PYTHONPATH=.
CKPT="${CKPT:-checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt}"
TAG="${TAG:?}"
OPT="${OPT:-}"
CONS_OPT="${CONS_OPT:-kcache,fastio,cl,graph${OPT:+,$OPT}}"
REGS="${REGS:-nr2,nr10,nr100,nr300,bh10,bh100,bh300,bh1000}"
STEPS="${STEPS:-10000}"
BASE="${BASE:-results/scatter_baseline_v2.json}"
mkdir -p results
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
[ -z "${TESTS:-}" ] || python scripts/scatter_fix_tests.py --ckpt "$CKPT"
python scripts/scatter_regress.py run --ckpt "$CKPT" --out "results/${TAG}_regress.json" ${OPT:+--opt $OPT}
[ ! -f "$BASE" ] || python scripts/scatter_regress.py compare --baseline "$BASE" --candidate "results/${TAG}_regress.json" || true
if [ -z "${NO_CONS:-}" ]; then
  python scripts/scatter_conservation.py probe --ckpt "$CKPT" --opt "$CONS_OPT" --out-dir "results/cons_${TAG}"
  python scripts/scatter_conservation.py run --ckpt "$CKPT" --opt "$CONS_OPT" --steps "$STEPS" --regimes "$REGS" --out-dir "results/cons_${TAG}"
fi
