#!/bin/bash
#SBATCH --job-name=augur-scatter-cl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=augur-scatter-cl-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_cl"
export PYTHONPATH=.
E="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
BASE=results/baseline_v2_regress.json
ITERS="${ITERS:-16000}"
TAG="${TAG:-cl}"
MODES="${MODES:-truth relabel}"
EXTRA="${EXTRA:-}"
mkdir -p results/scatter_cl checkpoints/scatter_cl
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
if [ -n "${CHECKS:-}" ]; then
  python -m pytest -q -x tests/test_scatter_cl.py 2>&1 | tail -20
fi
if [ -n "${BASEEVAL:-}" ]; then
  python scripts/eval_scatter_cl.py --ckpt "$E" --out results/scatter_cl/base_past.json || echo "BASE EVAL FAILED"
fi
for mode in $MODES; do
  name="${TAG}_${mode}"
  ck="checkpoints/scatter_cl/$name.pt"
  python scripts/train_scatter_cl.py --mode "$mode" --name "$name" --init-ckpt "$E" --iters "$ITERS" $EXTRA || echo "TRAIN FAILED $name"
  python scripts/eval_scatter_cl.py --ckpt "$ck" --out "results/scatter_cl/${name}_past.json" || echo "PAST EVAL FAILED $name"
  python scripts/scatter_regress.py run --ckpt "$ck" --out "results/scatter_cl/${name}_regress.json" || echo "REGRESS FAILED $name"
  python scripts/scatter_regress.py compare --baseline "$BASE" --candidate "results/scatter_cl/${name}_regress.json" || true
done
if [ -f "results/scatter_cl/${TAG}_truth_regress.json" ] && [ -f "results/scatter_cl/${TAG}_relabel_regress.json" ]; then
  echo "=== relabel vs same-schedule control"
  python scripts/scatter_regress.py compare --baseline "results/scatter_cl/${TAG}_truth_regress.json" --candidate "results/scatter_cl/${TAG}_relabel_regress.json" || true
fi
echo "CL DONE $TAG"
