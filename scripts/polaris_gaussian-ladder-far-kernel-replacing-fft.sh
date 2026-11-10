#!/bin/bash
#SBATCH --job-name=augur-gaussian-ladder-far-kernel-replacing-fft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SUPERVISOR_REPLACES_WITH_SLOT_TAIL_JOB_ID
#SBATCH --output=augur-gaussian-ladder-far-kernel-replacing-fft-%j.log
# Stage 1 (fit the ladder to the frozen learned kernel, then a short GPU rollout check). Pre-registration: docs/debugging/architecture-push-prereg/gaussian-ladder-far-kernel-replacing-fft.md.
# Estimate 0.5 GPU-h (spec); CPU fits take seconds, the rollout check is 3 eval seeds x 48 scenes x 20 ticks x 2 models (rate not logged for this model pair; untested). Cells are saved in the json and skipped on rerun: sbatch again to resume.
# Needs $HOME/augur-architecture-push-stage-one-code.tgz (scripts/build_architecture_push_stage_one_code_tarball.sh), $HOME/augur_nfr/kern/E_frozen.pt and $HOME/augur_nfr/ckpt/main_s4739.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
CODE=$HOME/augur-architecture-push-stage-one-code.tgz
W=$HOME/augur-gaussian-ladder-far-kernel-replacing-fft
E=$HOME/augur_nfr/kern/E_frozen.pt
M=$HOME/augur_nfr/ckpt/main_s4739.pt
for f in $CODE $E $M; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
mkdir -p $W/results || exit 2
SUM=$(md5sum $CODE | cut -d' ' -f1)
[ -e $W/.extracted-$SUM ] || { tar xzf $CODE -C $W && touch $W/.extracted-$SUM; } || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=$W
for f in scripts/scatter_ladder.py scripts/scatter_ladder_stage1.py scripts/scatter_multi.py scripts/scatter_multi_eval.py scripts/score_architecture_push_stage_one.py; do
  [ -s $f ] || { echo "MISSING $f in tarball"; exit 2; }
done
grep -q "def ppe" scripts/scatter_field.py || { echo "STALE scatter_field.py: no ppe"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_ladder.py scripts/scatter_ladder_stage1.py $E $M
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/gaussian-ladder-far-kernel-replacing-fft-stage-one.json
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/scatter_ladder_stage1.py --e-ckpt $E --main-ckpt $M --out $O --seeds 9100,9200,9300 >> ladder-stage-one.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "run exit $RC"; tail -20 ladder-stage-one.log; exit $RC; }
[ -s $O ] && [ -e $O.done ] || { echo "MISSING OUTPUT $O or .done"; exit 1; }
for c in fit:E:pow2 fit:main:pow1 fit:main:pow2 fit:main:pow3 fit:main:yuk8 fit:main:yuk128 fit:main:lj tiled timing rollout:9100,9200,9300; do
  grep -q "\"$c\"" $O || { echo "MISSING CELL $c"; exit 1; }
done
python scripts/score_architecture_push_stage_one.py ladder --in $O --out results/gaussian-ladder-far-kernel-replacing-fft-stage-one-score.json || exit 1
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
