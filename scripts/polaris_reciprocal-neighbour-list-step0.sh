#!/bin/bash
#SBATCH --job-name=augur-reciprocal-neighbour-list-step0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-reciprocal-neighbour-list-step0-%j.log
# Feasibility probe, audit row 32 (close-pair proposal R4 step 0): frozen model, union neighbour list vs top-16 list on nn<1 force error, 3 families x 16 scenes at N=1000, seed 4738.
# Pre-registration: docs/debugging/director-feasibility-2026-11-07.md. Inference only, estimate under 0.2 GPU-h (not timed).
# Needs in $HOME/augur_nfr: kern/E_frozen.pt, scripts/scatter_field.py, scripts/scatter_scaling_ops.py, scripts/pm_baseline_force_error.py and the new
# scripts/reciprocal_neighbour_list_step0.py (upload it there). Output is saved per family and the job is skipped if the result exists: sbatch again to resume.
# Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
DEADLINE=1794070800
REMAIN=$((DEADLINE-$(date +%s)))
[ $REMAIN -le 300 ] && { echo "DEADLINE 2026-11-07 09:00 PST: $REMAIN s left, not starting"; exit 3; }
dlbg() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; exit 124; }
  exec timeout -k 120 -s TERM $left "$@"
}
PID=
cd $HOME/augur_nfr || { echo "MISSING $HOME/augur_nfr"; exit 2; }
export PYTHONPATH=.
for f in kern/E_frozen.pt scripts/scatter_field.py scripts/scatter_scaling_ops.py scripts/pm_baseline_force_error.py scripts/reciprocal_neighbour_list_step0.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/reciprocal_neighbour_list_step0
mkdir -p $O
T0=$(date +%s)
if [ -s $O/step0.json ] && grep -q '^DONE' $O/step0.log 2>/dev/null; then echo "already done"; echo ALLDONE; exit 0; fi
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
dlbg python scripts/reciprocal_neighbour_list_step0.py --ckpt kern/E_frozen.pt --out $O/step0.json --seed 4738 > $O/step0.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $((DEADLINE-$(date +%s))) -le 300 ] && [ $RC -ne 0 ] && { echo "DEADLINE_STOP: unfinished"; exit 4; }
[ $RC -eq 0 ] || { echo "step0 exit $RC"; tail -20 $O/step0.log; exit $RC; }
grep -q '^DONE' $O/step0.log && [ -s $O/step0.json ] || { echo "MISSING OUTPUT $O/step0.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
