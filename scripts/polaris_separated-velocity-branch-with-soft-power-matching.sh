#!/bin/bash
#SBATCH --job-name=augur-input-group-work-split-probe
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
#SBATCH --output=augur-input-group-work-split-probe-%j.log
##SBATCH --dependency=afterany:<TAIL_JOB_ID_OF_SLOT>
# separated-velocity-branch-with-soft-power-matching, stage 1 SPLIT ONLY (the power-matching term is not run; inference, no training, 1 GPU). Work done by each input group on the replay checkpoint (job 4089), seeds 9100/9200/9300.
# Estimate about 1.2 GPU-h (1 rollout plus 2 estimators x 5 group re-evaluations per tick, x 3 seeds x 24 scenes; untimed assumption, the 1.5 h time limit is the cap). Code tarball: bash scripts/build_tier0_stage1_probes_code_tarball.sh tier0_stage1_probes_code.tgz, upload to $HOME.
# Usage: remove one # from the dependency line and fill the slot tail, then sbatch. Requeue-safe: skips if the result json exists.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_input_group_work_split_probe
mkdir -p $W && cd $W && tar xzf $HOME/tier0_stage1_probes_code.tgz
export PYTHONPATH=$W
O=results/input_group_work_split_probe
mkdir -p $O
CK=$HOME/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt
for f in scripts/input_group_work_split_probe.py scripts/tier0_stage1_probe_common.py scripts/tier0_tiled_closed_loop.py scripts/eval_tier0_tiled_closed_loop.py scripts/train_tiered.py \
  scripts/exp12_hier.py scripts/scatter_field.py model/tiered/__init__.py $CK; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$O/input_group_work_split_probe.json
[ -s $OUT ] && { echo "already done: $OUT"; exit 0; }
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/input_group_work_split_probe.py --ckpt replay=$CK --out $OUT &
PID=$!
wait $PID
rc=$?
[ $rc -eq 0 ] && [ -s $OUT ] || { echo "FAILED rc=$rc"; exit 1; }
echo "OK $OUT"
