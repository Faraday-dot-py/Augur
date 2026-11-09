#!/bin/bash
#SBATCH --job-name=augur-pair-term-adaptive-depth-fix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:7193
#SBATCH --output=augur-pair-term-adaptive-depth-fix-%j.log
# Feasibility test: deeper adaptive layers instead of the all-bodies exact fallback, collapsed Plummer N=1e6 and 3e6 plus the 3e6 reference cloud, seed 4738.
# Pre-registration: docs/debugging/experiment-log.md (2026-11-08 pair-term-adaptive-depth-fix). Inference only, estimate under 20 min (not timed; collapsed 3e6 default variant alone is one 222 s call).
# Needs: $HOME/augur_probe_deep/scripts/{scatter_field.py,pair_term_adaptive_depth_test.py,scatter_scaling_ops.py,pm_baseline_force_error.py,scatter_field_head.py} and kern/E_frozen.pt.
# Per-cell results are written per cell and skipped on rerun: sbatch again to resume.
set -uo pipefail
export PYTHONUNBUFFERED=1
D=$HOME/augur_probe_deep
ln -sfn $HOME/augur_nfr/kern $D/kern; cd $D || { echo "MISSING $D"; exit 2; }
export PYTHONPATH=.
for f in kern/E_frozen.pt scripts/scatter_field.py scripts/scatter_field_head.py scripts/scatter_scaling_ops.py scripts/pm_baseline_force_error.py scripts/pair_term_adaptive_depth_test.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q adaptdeep scripts/scatter_field.py || { echo "STALE scatter_field.py: no adaptdeep"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/pair_term_adaptive_depth_fix
mkdir -p $O
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
run() {
  local tag=$1; shift
  [ -s $O/$tag.json ] && { echo "skip $tag"; return 0; }
  python scripts/pair_term_adaptive_depth_test.py --ckpt kern/E_frozen.pt --device cuda --out $O/$tag.json "$@" >> $O/$tag.log 2>&1 &
  PID=$!
  wait $PID
  local rc=$?
  [ $rc -eq 0 ] || { echo "$tag exit $rc"; tail -20 $O/$tag.log; exit $rc; }
  grep -q '^DONE' $O/$tag.log && [ -s $O/$tag.json ] || { echo "MISSING OUTPUT $O/$tag.json"; exit 1; }
}
run collapsed-1e6 --n 1000000 --core-scale 0.25 --reps 3 --head-module scripts/scatter_field_head.py
run reference-3e6 --n 3000000 --core-scale 1.0 --reps 3
run collapsed-3e6 --n 3000000 --core-scale 0.25 --reps 3 --untimed default
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
