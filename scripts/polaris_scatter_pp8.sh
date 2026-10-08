#!/bin/bash
#SBATCH --job-name=augur-scatter-pp8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-pp8-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur_pp8"
mkdir -p "$W"
cp -r "$HOME/augur/model" "$W/"
mkdir -p "$W/scripts"
cp "$HOME/augur/scripts/"*.py "$W/scripts/"
cp "$HOME/augur/"*.py "$W/" 2>/dev/null || true
cp "$HOME/augur_pp8_up/"*.py "$W/scripts/"
mkdir -p "$W/results/scatter_pp8" "$W/checkpoints/scatter_pp8"
cp -n "$HOME/augur/results/scatter_baseline.json" "$W/results/" || true
cd "$W"
export PYTHONPATH=.
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
MODE="${MODE:-train}"
ITERS="${ITERS:-4000}"
MODELS="${MODELS:-ctl,pair,prior,pp8,cf}"
EBASE="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
case "$MODE" in
  train) python scripts/scatter_prior_pp8.py --models "$MODELS" --iters "$ITERS" --out "${OUT:-results/scatter_pp8}" --ckpt checkpoints/scatter_pp8 ${EXTRA:-} ;;
  frozen) python scripts/scatter_prior_pp8.py --models ctl --eval-ckpt "$EBASE" --tag _frozenE --out results/scatter_pp8
          python scripts/scatter_regress.py run --ckpt "$EBASE" --out results/scatter_pp8/regress_frozenE.json
          python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate results/scatter_pp8/regress_frozenE.json || true ;;
  regress) for m in ${MODELS//,/ }; do
             V=ms_kp_pot_v_g128
             case "$m" in prior) V=ms_kp_pot_v_g128_prior;; pp8) V=ms_kp_pot_v_g128_pp8;; pair) V=ms_kp_pot_v_g128_prior_pp8;; cf) V=cf;; esac
             python scripts/scatter_regress.py run --ckpt "checkpoints/scatter_pp8/$m.pt" --variant "$V" --out "results/scatter_pp8/regress_${m}_${ITERS}.json"
             python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate "results/scatter_pp8/regress_${m}_${ITERS}.json" || true
           done ;;
  scale) python scripts/scatter_pp8_scale.py --variant "$VARIANT" ${CKPT:+--ckpt "$CKPT"} --out "results/scatter_pp8/scale_${TAG}.json" ;;
esac
echo "[$(date -Iseconds)] done"
