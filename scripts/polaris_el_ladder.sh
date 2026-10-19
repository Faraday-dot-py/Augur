#!/bin/bash
#SBATCH --job-name=el-ladder
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=el-ladder-%j.log
# Euclid-loss energy-fix ladder vs frozen E. Resumable: finished arms are skipped (done markers), train() resumes from the 100-it full-state
# checkpoint in $ROOT/checkpoints, SIGTERM saves a checkpoint (handler in train_scatter_field.py) and the wrapper re-runs on requeue.
set -uo pipefail
export PYTHONUNBUFFERED=1
ROOT=$HOME/augur-el
mkdir -p $ROOT
cd $ROOT
[ -f scripts/ladder_summary.py ] || tar xf $HOME/el_code.tar
mkdir -p results/el_eval checkpoints
export PYTHONPATH=.
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
LAM=${LAM:-5}
ITERS=${ITERS:-10000}
trap 'echo "SIGTERM"; kill -TERM $(jobs -p) 2>/dev/null; wait; exit 143' TERM

train() {
  tag=$1; seed=$2; shift 2
  python scripts/train_scatter_field.py --exp E --variant ms_kp_pot_v_g128 --mode train --init-ckpt $E --train 1500 --steps 50 --iters $ITERS --batch 16 --k-start 20 --k-end 20 --lr 1e-4 --device cuda --seed $seed --rloss euclid --out-dir results/el_$tag --ckpt-dir checkpoints/el_$tag "$@" 2>&1 | grep -v Warning &
  wait $!
}
evalall() {
  tag=$1; ck=$2
  [ -f results/el_eval/done_$tag ] && return
  python scripts/scatter_losstest.py --setup warm --arm sq --tag _$tag --eval-ckpt $ck --out results/el_eval --ckpt checkpoints/el_eval --iters 0 2>&1 | grep -v Warning
  for r in 1 2 3; do python scripts/scatter_regress.py run --ckpt $ck --out results/el_eval/regress_${tag}_r$r.json 2>&1 | grep -E "^cons|MEAN|Error|Traceback"; done
  echo "=== compare $tag (r1 vs scatter_baseline)"
  python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate results/el_eval/regress_${tag}_r1.json
  touch results/el_eval/done_$tag
}
arm() {
  name=$1; shift
  for seed in 4738 4739; do
    tag=${name}_s$seed
    ck=checkpoints/el_$tag/E_ms_kp_pot_v_g128.pt
    if [ ! -f results/el_$tag/E_ms_kp_pot_v_g128.json ]; then
      echo "[$(date -Iseconds)] train $tag"
      train $tag $seed "$@"
      [ -f results/el_$tag/E_ms_kp_pot_v_g128.json ] || { echo "TRAIN FAILED $tag"; return 1; }
      echo "[$(date -Iseconds)] trained $tag"
    fi
    evalall $tag $ck
  done
  python scripts/ladder_summary.py ${name}_s4738 ${name}_s4739
}

echo "=== frozen E reference (3 regress runs, same code)"
evalall frozenE $E
echo "=== Run 1: heads-only (kmlp,ppmlp), Euclid loss"
arm r1 --train-only kmlp,ppmlp
if python scripts/ladder_summary.py --gate r1_s4738 r1_s4739 > /dev/null; then echo "R1 PASSES BOTH CRITERIA: ladder stops"; exit 0; fi
echo "=== Run 2: full fine-tune + L2-SP lambda $LAM"
arm r2 --l2sp $LAM
if python scripts/ladder_summary.py --gate r2_s4738 r2_s4739 > /dev/null; then echo "R2 PASSES BOTH CRITERIA: ladder stops"; exit 0; fi
echo "=== Run 3: Run 2 + nr100/bh300 replay (25% of iterations)"
arm r3 --l2sp $LAM --replay-nr100 128 --replay-bh300 96 --replay-frac 0.25
python scripts/ladder_summary.py frozenE r1_s4738 r1_s4739 r2_s4738 r2_s4739 r3_s4738 r3_s4739
echo "[$(date -Iseconds)] LADDER DONE"
