#!/bin/bash
#SBATCH --job-name=augur-interior-force-offset-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-interior-force-offset-probe-%j.log
# Feasibility probe, audit row 10: attributes the ~1.55x interior force offset at 1e6 bodies (learned vs exact point sum) to mass density, pair term or grid kernel.
# Pre-registration: docs/debugging/director-feasibility-2026-11-07.md. Seed 4738. Inference only, estimate under 1 GPU-h (exact fp64 sum at N=1e5, sampled targets at 1e6; not timed).
# Needs in $HOME/augur_nfr: kern/E_frozen.pt, scripts/scatter_field.py, scripts/scatter_scaling_ops.py, scripts/pm_baseline_force_error.py (from earlier jobs) and the new
# scripts/interior_force_offset_probe.py (upload it there). Per-cell results are saved after each cell and skipped on rerun: sbatch this script again to resume.
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
for f in kern/E_frozen.pt scripts/scatter_field.py scripts/scatter_scaling_ops.py scripts/pm_baseline_force_error.py scripts/interior_force_offset_probe.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/interior_force_offset_probe
mkdir -p $O
T0=$(date +%s)
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
dlbg python scripts/interior_force_offset_probe.py --ckpt kern/E_frozen.pt --out $O/probe.json --seed 4738 >> $O/probe.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $((DEADLINE-$(date +%s))) -le 300 ] && [ $RC -ne 0 ] && { echo "DEADLINE_STOP: unfinished, per-cell results saved"; exit 4; }
[ $RC -eq 0 ] || { echo "probe exit $RC"; tail -20 $O/probe.log; exit $RC; }
grep -q '^DONE' $O/probe.log && [ -s $O/probe.json ] || { echo "MISSING OUTPUT $O/probe.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
