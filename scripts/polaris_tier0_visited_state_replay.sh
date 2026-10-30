#!/bin/bash
#SBATCH --job-name=augur-tier0-visited-state-label-replay
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-tier0-visited-state-label-replay-%j.log
# visited-state-label-replay-tier-0: fine-tune the e44b796 tiled closed-loop tier 0 with 3 rounds x 1500 iterations; each round rolls 24 fresh training scenes 100 ticks without
# gradient, labels the states at ticks 20, 25, ..., 100 with the exact force (training scenes only) and adds a one-step force regression at those states to the usual closed-loop recipe.
# Probes at 1500 and 3000 (end of rounds 1 and 2): kill if dE@40 above 1.0x / 0.85x the starting model's or err gates (as pair-potential-by-separation script).
# Requeue-safe (full state every 250 iterations and on SIGTERM; a requeue rebuilds the current round's replay set from the resumed model). Usage: SEED=4738 sbatch --export=ALL,SEED=4738 this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED=${SEED:-4738}
W=$HOME/augur_tier0_visited_state_replay_$SEED
mkdir -p $W && cd $W && tar xzf $HOME/visited_replay_code.tgz
export PYTHONPATH=$W
O=results/tier0_visited_state_label_replay
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt
REF=$HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json $E $INIT $REF $HOME/augur_tiered_c2_tier0_matched_60k.pt $HOME/t0n/data4; do [ -e $f ] || { echo "MISSING $f"; exit 1; }; done
TAG=tier0-visited-state-label-replay-seed$SEED
T0=$(date +%s)
[ -f $O/tests.done ] || { python scripts/tier0_tiled_closed_loop_tests.py --device cpu > $O/tests.log 2>&1 && touch $O/tests.done || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }; }
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/$TAG.TRAIN_DONE ] && [ ! -f $O/$TAG.EARLY_KILL ]; then
  python scripts/train_tier0_tiled_closed_loop.py --init $INIT --data $HOME/t0n/data4 --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $O \
    --exchange halo --iters 4500 --lr 1e-4 --k-start 16 --k-end 28 --replay-rounds 3 --seed $SEED --val-seed ${VAL_SEED:-4739} --probe-at 1500 3000 \
    --gate1-err 1.10 --gate1-force 1.10 --gate2-err 1.10 --gate2-bl 1.25 --gate1-dE 1.0 --gate2-dE 0.85 --resume >> $O/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "train exit $RC"; exit $RC; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
[ -f $O/$TAG.EARLY_KILL ] && { echo "EARLY_KILL: $(cat $O/$TAG.EARLY_KILL)"; exit 0; }
[ -f $O/$TAG.TRAIN_DONE ] || { echo "training incomplete"; exit 0; }
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/${TAG}_eval.json ] || python scripts/eval_tier0_tiled_closed_loop.py --ckpt $O/$TAG.pt --exchange halo --out $O/${TAG}_eval.json > $O/$TAG.eval.log 2>&1
[ -f $O/${TAG}_hext.json ] || python scripts/hetero_ext_checks.py --ckpt $O/$TAG.pt --tag $TAG --out $O/${TAG}_hext.json > $O/$TAG.hext.log 2>&1
python scripts/score_tier0_visited_state_replay.py $O/${TAG}_eval.json $REF --out $O/${TAG}_score.json
python scripts/tier0_energy_metric_audit.py --tier0 $E --m60k $HOME/augur_tiered_c2_tier0_matched_60k.pt --new $O/$TAG.pt --out $O/${TAG}_energy_audit.json > $O/$TAG.audit.log 2>&1
python scripts/tier0_drift_diagnoses.py --new $O/$TAG.pt --out $O/${TAG}_drift_diagnoses.json > $O/$TAG.diag.log 2>&1
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
