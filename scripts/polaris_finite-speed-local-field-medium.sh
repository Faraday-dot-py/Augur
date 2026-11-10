#!/bin/bash
#SBATCH --job-name=augur-finite-speed-local-field-medium
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-finite-speed-local-field-medium-%j.log
# --dependency=afterany:TAIL_JOB_ID  (placeholder: the supervisor passes it on the sbatch command line from the slot tail in docs/debugging/gpu-slot-chains.md; never submit without it)
# CURIOSITY run (no pre-registered prediction, no pass/kill verdict), hard cap 1 GPU-h = the --time limit above. Four propagation speeds (1, 2, 4, 8 field substeps per tick), 480 s of training each
# (32 min total), then observations at N = 2, 20, 1000 on seeds 9100/9200/9300 and the retarded-force front check at N = 2 and 20. Results per speed are saved as they finish and skipped on rerun.
# Runtime estimate (untimed, no logged rate for this model): 32 min training budget plus under 10 min of evaluation, total under 45 min.
# Needs $HOME/finite_speed_local_field_medium_code.tgz (scripts/build_architecture_push_stage1_code_tarball.sh).
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur-finite-speed-local-field-medium
TGZ=$HOME/finite_speed_local_field_medium_code.tgz
[ -s $TGZ ] || { echo "MISSING $TGZ"; exit 2; }
mkdir -p $W || exit 2
tar xzf $TGZ -C $W || { echo "TAR EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=.
for f in scripts/finite_speed_local_field_medium.py scripts/scatter_scaling.py scripts/scatter_scaling_ops.py model/local_field_medium.py model/__init__.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
md5sum scripts/finite_speed_local_field_medium.py model/local_field_medium.py scripts/scatter_field.py
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=results/finite_speed_local_field_medium
mkdir -p $OUT
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/finite_speed_local_field_medium.py --device cuda --nsubs 1,2,4,8 --iters 100000 --train-seconds 480 --out $OUT >> $OUT/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "finite_speed_local_field_medium exit $RC"; tail -20 $OUT/run.log; exit $RC; }
[ -s $OUT/results.json ] || { echo "MISSING OUTPUT $OUT/results.json"; exit 1; }
for n in 1 2 4 8; do
  grep -q "\"nsub$n\"" $OUT/results.json || { echo "MISSING nsub$n in results.json"; exit 1; }
done
echo "curiosity run complete: observations only, no verdict"
