#!/bin/bash
#SBATCH --job-name=augur-dense-region-exact-force-ceiling-probe
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
#SBATCH --output=augur-dense-region-exact-force-ceiling-probe-%j.log
##SBATCH --dependency=afterany:<TAIL_JOB_ID_OF_SLOT>
# nested-refinement-levels-on-the-scatter-field-with-smooth-level-blend, stage 1 ceiling probe only (inference, no training, 1 GPU). Exact force inside dense regions of the replay checkpoint (job 4089), seeds 9100/9200/9300.
# Estimate about 1.0 GPU-h (4 arms x 3 seeds x 24 scenes x 100 ticks; breakdown jobs logged about 0.3 GPU-h per full-label rollout set; untimed assumption). Code tarball: bash scripts/build_tier0_stage1_probes_code_tarball.sh tier0_stage1_probes_code.tgz, upload to $HOME.
# Usage: remove one # from the dependency line and fill the slot tail, then sbatch. Requeue-safe: skips if the result json exists.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_dense_region_exact_force_ceiling_probe_harness_fix
mkdir -p $W && cd $W && tar xzf $HOME/tier0_stage1_probes_code.tgz
export PYTHONPATH=$W
O=results/dense_region_exact_force_ceiling_probe
mkdir -p $O
CK=$HOME/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt
for f in scripts/dense_region_exact_force_ceiling_probe.py scripts/tier0_stage1_probe_common.py scripts/replay_mechanism_force_error_breakdown.py scripts/tier0_tiled_closed_loop.py \
  scripts/eval_tier0_tiled_closed_loop.py scripts/train_tiered.py scripts/exp12_hier.py scripts/scatter_field.py model/tiered/__init__.py $CK; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$O/dense_region_exact_force_ceiling_probe.json
[ -s $OUT ] && { echo "already done: $OUT"; exit 0; }
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/dense_region_exact_force_ceiling_probe.py --ckpt replay=$CK --out $OUT &
PID=$!
wait $PID
rc=$?
[ $rc -eq 0 ] && [ -s $OUT ] || { echo "FAILED rc=$rc"; exit 1; }
echo "OK $OUT"
