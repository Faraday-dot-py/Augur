#!/bin/bash
#SBATCH --job-name=augur-tiered-sharpness-aware-minimisation-adam
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
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-tiered-sharpness-aware-minimisation-adam-%j.log
# Job 4060 recipe (matched-60k init, tiled closed-loop halo fine-tune, 5000 iterations, default probes and seeds) with only the optimizer changed: Adam -> sharpness-aware minimisation around Adam
# (--optimizer sam-adam, fixed radius --sam-rho 0.05, no sweep; docs/debugging/optimizer-sweep-proposal.md section 4.3). Each iteration is two forward-backward passes on the same batch (generator state reset),
# the second at the weights moved by rho along the normalised first gradient; weights are restored before the clipped Adam step. Estimate 2.8 GPU-h: 2 x 0.97 s/it (job 4818 fine-tune stage, 4831 s for 5000 it) x 5000
# = 2.7 h + eval and hext about 0.1 h; SOAP is not included. Requeue-safe: trainer resumes from <tag>.pt (full state every 250 iterations and on SIGTERM); finished outputs are skipped.
# Code tarball: bash scripts/build_replay_code_tarball.sh tiered_overnight_code.tgz, upload to $HOME. Usage: sbatch this.sh
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
W=$HOME/augur_tiered_sharpness_aware_minimisation_adam
mkdir -p $W && cd $W && tar xzf $HOME/tiered_overnight_code.tgz
export PYTHONPATH=$W
O=results/tiered_sharpness_aware_minimisation_adam
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur_tiered_c2_tier0_matched_60k.pt
D=$HOME/t0n/data4
TAG=tier0-tiled-closed-loop-halo-sam-adam
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json \
  scripts/score_tiered_sharpness_aware_minimisation_adam.py scripts/train_tier0_tiled_closed_loop.py scripts/eval_tier0_tiled_closed_loop.py scripts/hetero_ext_checks.py scripts/tier0_tiled_closed_loop_tests.py model/__init__.py $E $INIT $D; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
ls $D/hv3_s*.npz > /dev/null || { echo "MISSING harvest shards"; exit 2; }
T0=$(date +%s)
[ -f $O/tests.done ] || { python scripts/tier0_tiled_closed_loop_tests.py --device cpu > $O/tests.log 2>&1 && touch $O/tests.done || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }; }
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/$TAG.TRAIN_DONE ] && [ ! -f $O/$TAG.EARLY_KILL ]; then
  dlbg python scripts/train_tier0_tiled_closed_loop.py --init $INIT --data $D --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $O \
    --exchange halo --iters 5000 --optimizer sam-adam --sam-rho 0.05 --resume >> $O/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  deadline_exit
  [ $RC -eq 0 ] || { echo "train exit $RC"; exit $RC; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
[ -f $O/$TAG.EARLY_KILL ] && { echo "EARLY_KILL: $(cat $O/$TAG.EARLY_KILL)"; exit 0; }
deadline_exit; [ -f $O/$TAG.TRAIN_DONE ] || { echo "fine-tune incomplete (SIGTERM save)"; exit 143; }
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/${TAG}_eval.json ] || dlrun python scripts/eval_tier0_tiled_closed_loop.py --ckpt $O/$TAG.pt --exchange halo --out $O/${TAG}_eval.json > $O/$TAG.eval.log 2>&1
[ -f $O/${TAG}_hext.json ] || dlrun python scripts/hetero_ext_checks.py --ckpt $O/$TAG.pt --tag $TAG --out $O/${TAG}_hext.json > $O/$TAG.hext.log 2>&1
dlrun python scripts/score_tiered_sharpness_aware_minimisation_adam.py $O/${TAG}_eval.json --out $O/${TAG}_score.json
for f in $O/${TAG}_eval.json $O/${TAG}_hext.json $O/${TAG}_score.json; do [ -s $f ] || { echo "MISSING OUTPUT $f"; exit 1; }; done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
