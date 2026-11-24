#!/bin/bash
#SBATCH --job-name=augur-neighbour-tile-force-error-predictability-probe
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-neighbour-tile-force-error-predictability-probe-%j.log
##SBATCH --dependency=afterany:<TAIL_JOB_ID_OF_SLOT>
# summary-token-exchange-between-neighbour-tiles, stage 1 (inference-only measurement; a fixed small MLP is fitted on fresh-scene rollouts of the frozen replay checkpoint, job 4089; 1 GPU). Eval seeds 9100/9200/9300, MLP train seeds 7100/7200/7300.
# Estimate about 0.5 GPU-h (6 plain rollouts plus 6 small MLP fits; spec stage-1 figure 0.5; untimed assumption). Code tarball: bash scripts/build_tier0_stage1_probes_code_tarball.sh tier0_stage1_probes_code.tgz, upload to $HOME.
# Usage: remove one # from the dependency line and fill the slot tail, then sbatch. Requeue-safe: skips if the result json exists.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_neighbour_tile_force_error_predictability_probe
mkdir -p $W && cd $W && tar xzf $HOME/tier0_stage1_probes_code.tgz
export PYTHONPATH=$W
O=results/neighbour_tile_force_error_predictability_probe
mkdir -p $O
CK=$HOME/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt
for f in scripts/neighbour_tile_force_error_predictability_probe.py scripts/tier0_stage1_probe_common.py scripts/replay_mechanism_force_error_breakdown.py scripts/tier0_tiled_closed_loop.py \
  scripts/eval_tier0_tiled_closed_loop.py scripts/train_tiered.py scripts/exp12_hier.py scripts/scatter_field.py model/tiered/__init__.py $CK; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$O/neighbour_tile_force_error_predictability_probe.json
[ -s $OUT ] && { echo "already done: $OUT"; exit 0; }
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/neighbour_tile_force_error_predictability_probe.py --ckpt replay=$CK --out $OUT &
PID=$!
wait $PID
rc=$?
[ $rc -eq 0 ] && [ -s $OUT ] || { echo "FAILED rc=$rc"; exit 1; }
echo "OK $OUT"
