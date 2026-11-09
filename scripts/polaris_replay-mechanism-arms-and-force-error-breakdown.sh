#!/bin/bash
#SBATCH --job-name=augur-replay-mechanism-arms-and-force-error-breakdown
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodelist=gpu01
#SBATCH --qos=student-pleiades
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-replay-mechanism-arms-and-force-error-breakdown-%j.log
# Curiosity mechanism diagnosis (no pass/kill): why visited-state label replay lowers tiled closed-loop tier-0 energy drift. Stages, one change per arm against the frozen base (job 4060, seed 4738), eval seeds 9100/9200/9300:
# 1 force-error breakdown before/after replay (4738: base vs job 4089; 4739: base job 4818 vs job 6168), 2 no-replay matched-compute, 3 late-ticks-only, 4 early-ticks-only, 5 one-round replay, 6 truth-state replay on the 4818 base,
# 7 summary of ratios, 8 final breakdown over every 4738-base arm. Each arm = job 4089 recipe (4500 it, lr 1e-4, k 16-28, halo, probes off) with one flag changed. Estimate 9 GPU-h (arm ~1.6 incl. eval from job 4089 1 h 32 min; breakdowns ~0.3 each, untimed assumption).
# Requeue-safe: trainer resumes from <tag>.pt (full state every 250 it and on SIGTERM); finished stages skipped. Code tarball: bash scripts/build_replay_mechanism_code_tarball.sh replay_mechanism_code.tgz, upload to $HOME. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_replay_mechanism
mkdir -p $W && cd $W && tar xzf $HOME/replay_mechanism_code.tgz
export PYTHONPATH=$W
O=results/replay_mechanism
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
D=$HOME/t0n/data4
CK4060=$HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt
CK4818=$HOME/augur_tier0_second_training_seed/results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739.pt
CK4089=$HOME/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt
CK6168=$HOME/augur_tiered_replay_second_base_fresh_seed_eval/results/tiered_replay_on_second_training_base/tier0-visited-state-label-replay-on-second-base-seed4739.pt
EV4060=results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json
EV4818=results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739_eval.json
EV4089=$HOME/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738_eval.json
EV6168=$HOME/augur_tiered_replay_second_base_fresh_seed_eval/results/tiered_replay_on_second_training_base/tier0-visited-state-label-replay-on-second-base-seed4739_eval.json
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json $EV4060 $EV4818 $EV4089 $EV6168 scripts/replay_mechanism_force_error_breakdown.py scripts/summarise_replay_mechanism_arms.py \
  scripts/train_tier0_tiled_closed_loop.py scripts/eval_tier0_tiled_closed_loop.py model/__init__.py $E $CK4060 $CK4818 $CK4089 $CK6168 $D; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
ls $D/hv3_s*.npz > /dev/null || { echo "MISSING harvest shards"; exit 2; }
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
stage() {
  echo "STAGE $* start $(( $(date +%s) - T0 )) s"
}
run() {
  "$@" &
  PID=$!
  wait $PID
  local rc=$?
  [ $rc -eq 0 ] || { echo "FAILED rc=$rc: $*"; exit $rc; }
}
breakdown() {
  local out=$1; shift
  [ -s $out ] || { run python scripts/replay_mechanism_force_error_breakdown.py --ckpt "$@" --out $out; }
}
arm() {
  local tag=$1 init=$2 seed=$3 val=$4; shift 4
  stage arm $tag
  if [ ! -f $O/$tag.TRAIN_DONE ]; then
    run python scripts/train_tier0_tiled_closed_loop.py --init $init --data $D --pattern "hv3_s*.npz" --tier0 $E --tag $tag --out-dir $O \
      --exchange halo --iters 4500 --lr 1e-4 --k-start 16 --k-end 28 --seed $seed --val-seed $val --probe-at --resume "$@" >> $O/$tag.train.log 2>&1
  fi
  [ -f $O/$tag.TRAIN_DONE ] || { echo "no TRAIN_DONE for $tag (EARLY_KILL or incomplete)"; exit 1; }
  [ -s $O/${tag}_eval.json ] || run python scripts/eval_tier0_tiled_closed_loop.py --ckpt $O/$tag.pt --exchange halo --out $O/${tag}_eval.json > $O/$tag.eval.log 2>&1
  [ -s $O/${tag}_eval.json ] || { echo "MISSING OUTPUT $O/${tag}_eval.json"; exit 1; }
}
stage breakdown-4738-4739
breakdown $O/force_error_breakdown_seed4738_base_vs_replay.json base4738=$CK4060 replay4738=$CK4089
breakdown $O/force_error_breakdown_seed4739_base_vs_replay.json base4739=$CK4818 replay4739=$CK6168
arm replay-mechanism-late-ticks-only $CK4060 4738 4739 --replay-rounds 3 --replay-start 60 --replay-steps 100
arm replay-mechanism-no-replay-matched-compute $CK4060 4738 4739 --replay-rounds 0
arm replay-mechanism-early-ticks-only $CK4060 4738 4739 --replay-rounds 3 --replay-start 5 --replay-steps 20
arm replay-mechanism-one-round-of-replay $CK4060 4738 4739 --replay-rounds 1
arm replay-mechanism-truth-state-replay-second-base $CK4818 4739 4740 --replay-rounds 3 --replay-source truth
stage summary
run python scripts/summarise_replay_mechanism_arms.py --base $EV4060 --replay $EV4089 --out $O/arms_summary_seed4738_base.json \
  late-ticks-only=$O/replay-mechanism-late-ticks-only_eval.json no-replay-matched-compute=$O/replay-mechanism-no-replay-matched-compute_eval.json \
  early-ticks-only=$O/replay-mechanism-early-ticks-only_eval.json one-round-of-replay=$O/replay-mechanism-one-round-of-replay_eval.json
run python scripts/summarise_replay_mechanism_arms.py --base $EV4818 --replay $EV6168 --out $O/arms_summary_seed4739_base.json \
  truth-state-replay-second-base=$O/replay-mechanism-truth-state-replay-second-base_eval.json
stage breakdown-all-arms
breakdown $O/force_error_breakdown_seed4738_all_arms.json base4738=$CK4060 replay4738=$CK4089 late=$O/replay-mechanism-late-ticks-only.pt noreplay=$O/replay-mechanism-no-replay-matched-compute.pt \
  early=$O/replay-mechanism-early-ticks-only.pt oneround=$O/replay-mechanism-one-round-of-replay.pt
for f in $O/force_error_breakdown_seed4738_base_vs_replay.json $O/force_error_breakdown_seed4739_base_vs_replay.json $O/arms_summary_seed4738_base.json $O/arms_summary_seed4739_base.json $O/force_error_breakdown_seed4738_all_arms.json; do
  [ -s $f ] || { echo "MISSING OUTPUT $f"; exit 1; }
done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
