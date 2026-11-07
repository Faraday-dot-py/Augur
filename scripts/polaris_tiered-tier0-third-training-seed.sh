#!/bin/bash
#SBATCH --job-name=augur-tiered-tier0-third-training-seed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --nodelist=gpu01
#SBATCH --qos=student-pleiades
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-tiered-tier0-third-training-seed-%j.log
# Third training seed of the tiled closed-loop tier 0, same recipe as job 4818 (second seed) at seed 4740: (1) matched-60k base from scratch (60000 iterations),
# (2) tiled closed-loop fine-tune (5000 iterations, no early-stop probes so the endpoint is always measured, validation seed 4741), (3) held-out eval seeds 9100/9200/9300 and scoring.
# Estimate 6.4 GPU-h (job 4818: 6:24:35 = base 18030 s + fine-tune stage 4831 s + eval). Requeue-safe: both trainers resume from <tag>.pt (full state every 250 iterations and on SIGTERM);
# finished outputs are skipped. Code tarball: bash scripts/build_replay_code_tarball.sh tiered_overnight_code.tgz, upload to $HOME. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
DEADLINE=1794070800
REMAIN=$((DEADLINE-$(date +%s)))
[ $REMAIN -le 300 ] && { echo "DEADLINE 2026-11-07 09:00 PST: $REMAIN s left, not starting"; exit 3; }
past_deadline() { [ $((DEADLINE-$(date +%s))) -le 300 ]; }
deadline_exit() { past_deadline && { echo "DEADLINE_STOP: unfinished, state saved by SIGTERM handler"; exit 4; }; }
dlbg() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; exit 124; }
  exec timeout -k 120 -s TERM $left "$@"
}
dlrun() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; return 124; }
  timeout -k 120 -s TERM $left "$@"
  local rc=$?
  past_deadline && [ $rc -ne 0 ] && { echo "DEADLINE_STOP: $* rc=$rc"; exit 4; }
  return $rc
}
SEED=4740
VAL_SEED=4741
W=$HOME/augur_tiered_tier0_third_training_seed
mkdir -p $W && cd $W && tar xzf $HOME/tiered_overnight_code.tgz
export PYTHONPATH=$W
O=results/tier0_ext
F=results/tier0_third_training_seed
mkdir -p $O $F
E=$HOME/E_ms_kp_pot_v_g128.pt
D=$HOME/t0n/data4
BASE=tier0-matched-60k-iterations-seed$SEED
TAG=tier0-tiled-closed-loop-halo-base-seed$SEED
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json scripts/score_tier0_second_training_seed.py scripts/train_tier0_harvest.py \
  scripts/train_tier0_tiled_closed_loop.py scripts/eval_tier0_tiled_closed_loop.py scripts/hetero_ext_checks.py scripts/tier0_tiled_closed_loop_tests.py model/__init__.py $E $D; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
ls $D/hv3_s*.npz > /dev/null || { echo "MISSING harvest shards"; exit 2; }
T0=$(date +%s)
[ -f $F/tests.done ] || { python scripts/tier0_tiled_closed_loop_tests.py --device cpu > $F/tests.log 2>&1 && touch $F/tests.done || { echo "TESTS FAILED"; tail -20 $F/tests.log; exit 1; }; }
FB="ext=6,extapply,taper=8,bnd,halo=8"
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/${BASE}_eval.json ]; then
  dlbg python scripts/train_tier0_harvest.py --data $D --pattern "hv3_s*.npz" --replay 0.5 --lr 3e-4 --iters 60000 --t0 "$FB" --tag $BASE --out-dir $O --baseline $E --seed $SEED >> $O/$BASE.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  deadline_exit
  [ $RC -eq 0 ] || { echo "base train exit $RC"; exit $RC; }
  [ -f $O/${BASE}_eval.json ] || { echo "base training incomplete (SIGTERM save)"; exit 143; }
fi
echo "BASE_SECONDS $(( $(date +%s) - T0 ))"
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/hext_$BASE.json ] || dlrun python scripts/hetero_ext_checks.py --ckpt $O/$BASE.pt --tag $BASE --out $O/hext_$BASE.json > $O/$BASE.hext.log 2>&1
set +e
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $F/$TAG.TRAIN_DONE ]; then
  dlbg python scripts/train_tier0_tiled_closed_loop.py --init $O/$BASE.pt --data $D --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $F \
    --exchange halo --iters 5000 --seed $SEED --val-seed $VAL_SEED --probe-at --resume >> $F/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  deadline_exit
  [ $RC -eq 0 ] || { echo "fine-tune exit $RC"; exit $RC; }
  [ -f $F/$TAG.TRAIN_DONE ] || { echo "fine-tune incomplete (SIGTERM save)"; exit 143; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $F/${TAG}_eval.json ] || dlrun python scripts/eval_tier0_tiled_closed_loop.py --ckpt $F/$TAG.pt --exchange halo --out $F/${TAG}_eval.json > $F/$TAG.eval.log 2>&1
[ -f $F/${TAG}_hext.json ] || dlrun python scripts/hetero_ext_checks.py --ckpt $F/$TAG.pt --tag $TAG --out $F/${TAG}_hext.json > $F/$TAG.hext.log 2>&1
dlrun python scripts/score_tier0_second_training_seed.py $F/${TAG}_eval.json --out $F/${TAG}_score.json
for f in $O/${BASE}_eval.json $O/hext_$BASE.json $F/${TAG}_eval.json $F/${TAG}_hext.json $F/${TAG}_score.json; do [ -s $f ] || { echo "MISSING OUTPUT $f"; exit 1; }; done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
