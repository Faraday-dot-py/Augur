#!/bin/bash
#SBATCH --job-name=augur-sparse-code-unet-single-law
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SLOT_TAIL_PLACEHOLDER
#SBATCH --output=augur-sparse-code-unet-single-law-%j.log
# Stage 1 (0.7 GPU-h by spec; est 1.0-1.3 h: width 32 candidate 8000 its at 0.308 s/it = 2461 s times sparse top-k overhead, plus ranking 11 sites x 32 channels and ablation draws). Pre-registration: docs/debugging/architecture-push-prereg/sparse-code-unet-single-law.md.
# Frozen controls: job-7256 width8.pt and a width-32 reference chosen by the supervisor/user via REF32 (default job-7256 width32.pt = seed 4738, which had a loss spike; clean ones: ckpt_width32_spike/seed4739 and seed4741 width32.pt). Requeue-safe: candidate ckpt every 500 its, ranking json saved, finished json skipped.
set -uo pipefail
export PYTHONUNBUFFERED=1
TGZ=${TGZ:-$HOME/augur-architecture-push-single-law-readability-code.tgz}
CTRL=$HOME/augur-single-law-network-sizing/ckpt
REF32=${REF32:-$CTRL/width32.pt}
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
for f in $CTRL/width8.pt $REF32; do test -f $f || { echo MISSING $f; exit 2; }; done
echo REF32 $REF32
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/single_law_network_sizing.py scripts/single_law_readability.py scripts/single_law_readability_probes.py
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/sparse-code-unet-single-law
mkdir -p $O ckpt
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/single_law_readability.py --experiment sparse-code --out-dir $O --ckpt-dir ckpt --control-dir $CTRL --seed 4738 --ref32-ckpt $REF32  >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "run exit $RC"; tail -20 $O/run.log; exit $RC; }
grep -q '^ALL DONE' $O/run.log && [ -s $O/candidate-sparse-code-e50-s4738.json ] || { echo "MISSING OUTPUT $O/candidate-sparse-code-e50-s4738.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
