#!/bin/bash
#SBATCH --job-name=augur-tier0-second-training-seed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-tier0-second-training-seed-%j.log
# Second training seed of the tiled closed-loop tier 0: (1) matched-60k base from scratch at seed 4739 (recipe of job 4041, 60000 iterations, ~5.1 GPU-h),
# (2) the tiled closed-loop fine-tune of that base (recipe of job 4060, 5000 iterations, no early-stop probes so the endpoint is always measured, ~1.4 GPU-h),
# (3) held-out eval seeds 9100/9200/9300 and pre-registered scoring. Requeue-safe: both trainers resume from <tag>.pt (full state every 250 iterations and on SIGTERM);
# finished outputs are skipped. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED=4739
W=$HOME/augur_tier0_second_training_seed
mkdir -p $W && cd $W && tar xzf $HOME/second_seed_code.tgz
export PYTHONPATH=$W
O=results/tier0_ext
F=results/tier0_second_training_seed
mkdir -p $O $F
E=$HOME/E_ms_kp_pot_v_g128.pt
D=$HOME/t0n/data4
BASE=tier0-matched-60k-iterations-seed$SEED
TAG=tier0-tiled-closed-loop-halo-base-seed$SEED
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json scripts/score_tier0_second_training_seed.py model/__init__.py $E $D; do [ -e $f ] || { echo "MISSING $f"; exit 1; }; done
ls $D/hv3_s*.npz > /dev/null || { echo "MISSING harvest shards"; exit 1; }
T0=$(date +%s)
FB="ext=6,extapply,taper=8,bnd,halo=8"
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/${BASE}_eval.json ]; then
  python scripts/train_tier0_harvest.py --data $D --pattern "hv3_s*.npz" --replay 0.5 --lr 3e-4 --iters 60000 --t0 "$FB" --tag $BASE --out-dir $O --baseline $E --seed $SEED >> $O/$BASE.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "base train exit $RC"; exit $RC; }
  [ -f $O/${BASE}_eval.json ] || { echo "base training incomplete (SIGTERM save)"; exit 143; }
fi
echo "BASE_SECONDS $(( $(date +%s) - T0 ))"
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/hext_$BASE.json ] || python scripts/hetero_ext_checks.py --ckpt $O/$BASE.pt --tag $BASE --out $O/hext_$BASE.json > $O/$BASE.hext.log 2>&1
[ -f $O/tests.done ] || { python scripts/tier0_tiled_closed_loop_tests.py --device cpu > $O/tests.log 2>&1 && touch $O/tests.done || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }; }
set +e
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $F/$TAG.TRAIN_DONE ]; then
  python scripts/train_tier0_tiled_closed_loop.py --init $O/$BASE.pt --data $D --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $F \
    --exchange halo --iters 5000 --seed $SEED --val-seed 4740 --probe-at --resume >> $F/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "fine-tune exit $RC"; exit $RC; }
  [ -f $F/$TAG.TRAIN_DONE ] || { echo "fine-tune incomplete (SIGTERM save)"; exit 143; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $F/${TAG}_eval.json ] || python scripts/eval_tier0_tiled_closed_loop.py --ckpt $F/$TAG.pt --exchange halo --out $F/${TAG}_eval.json > $F/$TAG.eval.log 2>&1
[ -f $F/${TAG}_hext.json ] || python scripts/hetero_ext_checks.py --ckpt $F/$TAG.pt --tag $TAG --out $F/${TAG}_hext.json > $F/$TAG.hext.log 2>&1
python scripts/score_tier0_second_training_seed.py $F/${TAG}_eval.json --out $F/${TAG}_score.json
for f in $O/${BASE}_eval.json $O/hext_$BASE.json $F/${TAG}_eval.json $F/${TAG}_hext.json $F/${TAG}_score.json; do [ -s $f ] || { echo "MISSING OUTPUT $f"; exit 1; }; done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
