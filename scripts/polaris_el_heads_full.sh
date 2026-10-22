#!/bin/bash
#SBATCH --job-name=el-headsfull
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=el-headsfull-%j.log
# Heads-only (kmlp,ppmlp) Euclid-loss fine-tune of frozen E at the full 60k-it lr 5e-4 schedule. One seed per job: SEED=4738 or 4739.
# Resume: train() auto-resumes from the 100-it full-state checkpoint in $ROOT/checkpoints; SIGTERM saves a checkpoint; --requeue reruns this script.
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED=${SEED:-4738}
ITERS=${ITERS:-60769}
LOGEVERY=${LOGEVERY:-100}
NAME=${NAME:-hf}
ROOT=$HOME/augur-el-$NAME-$SEED
mkdir -p $ROOT
cd $ROOT
[ -f scripts/ladder_summary.py ] || tar xf $HOME/el_code.tar
mkdir -p results/el_eval checkpoints
export PYTHONPATH=.
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
trap 'echo "SIGTERM"; kill -TERM $(jobs -p) 2>/dev/null; wait; exit 143' TERM
tag=${NAME}_s$SEED
ck=checkpoints/el_$tag/E_ms_kp_pot_v_g128.pt
evalall() {
  t=$1; c=$2
  [ -f results/el_eval/done_$t ] && return
  python scripts/scatter_losstest.py --setup warm --arm sq --tag _$t --eval-ckpt $c --out results/el_eval --ckpt checkpoints/el_eval --iters 0 2>&1 | grep -v Warning
  for r in 1 2 3; do python scripts/scatter_regress.py run --ckpt $c --out results/el_eval/regress_${t}_r$r.json 2>&1 | grep -E "^cons|MEAN|Error|Traceback"; done
  touch results/el_eval/done_$t
}
if [ ! -f results/el_$tag/E_ms_kp_pot_v_g128.json ]; then
  echo "[$(date -Iseconds)] train $tag iters $ITERS"
  python scripts/train_scatter_field.py --exp E --variant ms_kp_pot_v_g128 --mode train --init-ckpt $E --train 1500 --steps 50 --iters $ITERS --batch 16 --k-start 20 --k-end 20 --lr 5e-4 --device cuda --seed $SEED --rloss euclid --train-only kmlp,ppmlp --log-every $LOGEVERY --out-dir results/el_$tag --ckpt-dir checkpoints/el_$tag 2>&1 | grep -v Warning &
  wait $!
  [ -f results/el_$tag/E_ms_kp_pot_v_g128.json ] || { echo "TRAIN FAILED/INTERRUPTED $tag"; exit 1; }
  echo "[$(date -Iseconds)] trained $tag"
fi
[ "${EVAL:-1}" = 1 ] || exit 0
[ "$SEED" = 4738 ] && evalall frozenE $E
evalall $tag $ck
echo "[$(date -Iseconds)] DONE $tag"
