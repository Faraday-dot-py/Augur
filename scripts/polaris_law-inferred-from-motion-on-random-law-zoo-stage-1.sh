#!/bin/bash
#SBATCH --job-name=augur-law-zoo-data-readiness-stage-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-law-zoo-data-readiness-stage-1-%j.log
# Stage 1 of law-inferred-from-motion-on-random-law-zoo: data and code readiness. Runs the CPU test files, generates 300 zoo laws x 2 scenes + 64 held-out-law scenes of exact truth, checks counts reconcile.
# No training and no probe (the spec defines none). Pre-registration: docs/debugging/architecture-push-prereg/law-inferred-from-motion-on-random-law-zoo.md.
# Estimate about 0.2 GPU-h (untimed; 11 chunks of 64 float64 scenes of 50 ticks). Requeue-safe: one chunk file per 64 scenes, skipped on rerun.
# Code tarball: bash scripts/build_architecture_push_stage1_code_tarball.sh, upload to $HOME. Supervisor adds --dependency at submission.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_architecture_push_stage1_lawzoo
T=$HOME/architecture_push_stage1_code.tgz
[ -s $T ] || { echo "MISSING $T"; exit 2; }
mkdir -p $W && cd $W && tar xzf $T || { echo "EXTRACT FAILED"; exit 2; }
export PYTHONPATH=$W
for f in scripts/law_zoo.py scripts/law_zoo_data_check.py scripts/scatter_multi.py scripts/scatter_far_general.py tests/test_law_zoo.py; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
md5sum scripts/law_zoo.py scripts/law_zoo_data_check.py
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/law_zoo_stage1
mkdir -p $O
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
[ -f $O/tests.done ] || { CUDA_VISIBLE_DEVICES= python -m pytest tests/test_law_zoo.py -q > $O/tests.log 2>&1 && touch $O/tests.done || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }; }
python scripts/law_zoo_data_check.py --out-dir $O --seed 4738 >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "exit $RC"; tail -20 $O/run.log; exit $RC; }
[ -s $O/data_summary.json ] || { echo "MISSING OUTPUT $O/data_summary.json"; exit 1; }
cat $O/data_summary.json
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
