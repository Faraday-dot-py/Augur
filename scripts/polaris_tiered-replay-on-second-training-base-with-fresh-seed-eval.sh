#!/bin/bash
#SBATCH --job-name=augur-tiered-replay-second-base-fresh-seed-eval
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
#SBATCH --output=augur-tiered-replay-second-base-fresh-seed-eval-%j.log
# Two parts. (b) Fresh held-out eval (seeds 12000/12100/12200, 24 scenes x 100 ticks) of the frozen job 4060 (training seed 4738) and job 4818 (training seed 4739) tiled checkpoints and of the
# tiled baseline on the same scenes; runs first because it is short and independent. (a) Visited-state label replay (job 4089 recipe: 3 rounds x 1500 iterations, lr 1e-4, k 16-28, early-stop probes off)
# started from the 4818 fine-tuned checkpoint, scored on seeds 9100/9200/9300 with the 4089 scorer against the 4818 eval. Estimate 2.0 GPU-h: replay about 1.55 (job 4089, 1 h 32 min for 4500 iterations
# at seed 4738) + about 0.3 evals and diagnostics, (b) about 0.15 (assumption from job 4818 eval stage, untimed). Requeue-safe: the trainer resumes from <tag>.pt (full state every 250 iterations and on
# SIGTERM; a requeue rebuilds the current round's replay set from the resumed model); finished outputs are skipped. Code tarball: bash scripts/build_replay_code_tarball.sh tiered_overnight_code.tgz, upload to $HOME.
# Checkpoint paths below are inferred from the job scripts of 4060/4089 and 4818, not verified on Polaris; the input check exits 2 if either is missing. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
DEADLINE=1794070800
REMAIN=$((DEADLINE-$(date +%s)))
[ $REMAIN -le 300 ] && { echo "DEADLINE 2026-11-07 09:00 PST: $REMAIN s left, not starting"; exit 3; }
past_deadline() { [ $((DEADLINE-$(date +%s))) -le 300 ]; }
deadline_exit() { past_deadline && { echo "DEADLINE_STOP: unfinished, state saved by SIGTERM handler"; exit 4; }; }
dlrun() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; return 124; }
  timeout -k 120 -s TERM $left "$@"
}
SEED=4739
VAL_SEED=4740
W=$HOME/augur_tiered_replay_second_base_fresh_seed_eval
mkdir -p $W && cd $W && tar xzf $HOME/tiered_overnight_code.tgz
export PYTHONPATH=$W
O=results/tiered_replay_on_second_training_base
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
D=$HOME/t0n/data4
CK4060=$HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt
CK4818=$HOME/augur_tier0_second_training_seed/results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739.pt
REF=results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739_eval.json
M60K=$HOME/augur_tiered_c2_tier0_matched_60k.pt
TAG=tier0-visited-state-label-replay-on-second-base-seed$SEED
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json $REF scripts/score_tier0_visited_state_replay.py scripts/score_tier0_fresh_seed_checkpoints.py \
  scripts/eval_tier0_fresh_seed_checkpoints.py scripts/eval_tier0_tiled_closed_loop.py scripts/train_tier0_tiled_closed_loop.py scripts/hetero_ext_checks.py scripts/tier0_energy_metric_audit.py \
  scripts/tier0_drift_diagnoses.py scripts/tier0_tiled_closed_loop_tests.py model/__init__.py $E $M60K $CK4060 $CK4818 $D; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
ls $D/hv3_s*.npz > /dev/null || { echo "MISSING harvest shards"; exit 2; }
T0=$(date +%s)
[ -f $O/tests.done ] || { python scripts/tier0_tiled_closed_loop_tests.py --device cpu > $O/tests.log 2>&1 && touch $O/tests.done || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }; }
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/fresh_seed_eval.json ] || dlrun python scripts/eval_tier0_fresh_seed_checkpoints.py --tier0 $E --ckpt seed4738-job4060=$CK4060 seed4739-job4818=$CK4818 --out $O/fresh_seed_eval.json > $O/fresh_seed_eval.log 2>&1
dlrun python scripts/score_tier0_fresh_seed_checkpoints.py $O/fresh_seed_eval.json --out $O/fresh_seed_score.json
echo "FRESH_EVAL_SECONDS $(( $(date +%s) - T0 ))"
set +e
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/$TAG.TRAIN_DONE ] && [ ! -f $O/$TAG.EARLY_KILL ]; then
  dlrun python scripts/train_tier0_tiled_closed_loop.py --init $CK4818 --data $D --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $O \
    --exchange halo --iters 4500 --lr 1e-4 --k-start 16 --k-end 28 --replay-rounds 3 --seed $SEED --val-seed $VAL_SEED --probe-at --resume >> $O/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  deadline_exit
  [ $RC -eq 0 ] || { echo "train exit $RC"; exit $RC; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
[ -f $O/$TAG.EARLY_KILL ] && { echo "EARLY_KILL: $(cat $O/$TAG.EARLY_KILL)"; exit 0; }
deadline_exit; [ -f $O/$TAG.TRAIN_DONE ] || { echo "training incomplete (SIGTERM save)"; exit 143; }
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/${TAG}_eval.json ] || dlrun python scripts/eval_tier0_tiled_closed_loop.py --ckpt $O/$TAG.pt --exchange halo --out $O/${TAG}_eval.json > $O/$TAG.eval.log 2>&1
[ -f $O/${TAG}_hext.json ] || dlrun python scripts/hetero_ext_checks.py --ckpt $O/$TAG.pt --tag $TAG --out $O/${TAG}_hext.json > $O/$TAG.hext.log 2>&1
dlrun python scripts/score_tier0_visited_state_replay.py $O/${TAG}_eval.json $REF --out $O/${TAG}_score.json
dlrun python scripts/tier0_energy_metric_audit.py --tier0 $E --m60k $M60K --new $O/$TAG.pt --out $O/${TAG}_energy_audit.json > $O/$TAG.audit.log 2>&1
dlrun python scripts/tier0_drift_diagnoses.py --new $O/$TAG.pt --out $O/${TAG}_drift_diagnoses.json > $O/$TAG.diag.log 2>&1
for f in $O/fresh_seed_eval.json $O/fresh_seed_score.json $O/${TAG}_eval.json $O/${TAG}_hext.json $O/${TAG}_score.json $O/${TAG}_energy_audit.json $O/${TAG}_drift_diagnoses.json; do [ -s $f ] || { echo "MISSING OUTPUT $f"; exit 1; }; done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
