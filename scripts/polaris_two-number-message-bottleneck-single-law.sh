#!/bin/bash
#SBATCH --job-name=augur-two-number-message-bottleneck-single-law
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SLOT_TAIL_PLACEHOLDER
#SBATCH --output=augur-two-number-message-bottleneck-single-law-%j.log
# Stage 1 (0.5-0.7 GPU-h by spec; est 0.7-0.8 h: width 8 candidate 8000 its at 0.276 s/it from job 7256 = 2211 s, plus control eval, probes). Pre-registration: docs/debugging/architecture-push-prereg/two-number-message-bottleneck-single-law.md.
# Frozen control: job-7256 width8.pt, seed 4738. Requeue-safe: candidate checkpoint every 500 its with implicit resume, finished json skipped; resubmit with a new dependency. TGZ built by the parent, uploaded by the supervisor.
set -uo pipefail
export PYTHONUNBUFFERED=1
TGZ=${TGZ:-$HOME/augur-architecture-push-single-law-readability-code.tgz}
CTRL=$HOME/augur-single-law-network-sizing/ckpt
W=$HOME/augur-single-law-readability
test -f $TGZ || { echo MISSING $TGZ; exit 2; }
mkdir -p $W && tar xzf $TGZ -C $W || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=$W
for f in scripts/single_law_readability.py scripts/single_law_readability_probes.py scripts/single_law_network_sizing.py scripts/scatter_field.py scripts/scatter_multi.py scripts/scatter_far_general.py model/__init__.py ; do
  test -f $f || { echo MISSING $f; exit 2; }
done
grep -q "bottleneck" scripts/scatter_multi.py || { echo "STALE scatter_multi.py"; exit 2; }
grep -q "sparse_k" scripts/scatter_field.py || { echo "STALE scatter_field.py"; exit 2; }
test -f $CTRL/width8.pt || { echo MISSING $CTRL/width8.pt; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/single_law_network_sizing.py scripts/single_law_readability.py scripts/single_law_readability_probes.py
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/two-number-message-bottleneck-single-law
mkdir -p $O ckpt
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/single_law_readability.py --experiment bottleneck --out-dir $O --ckpt-dir ckpt --control-dir $CTRL --seed 4738  >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "run exit $RC"; tail -20 $O/run.log; exit $RC; }
grep -q '^ALL DONE' $O/run.log && [ -s $O/candidate-bottleneck-e50-s4738.json ] || { echo "MISSING OUTPUT $O/candidate-bottleneck-e50-s4738.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
