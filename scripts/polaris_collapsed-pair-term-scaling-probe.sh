#!/bin/bash
#SBATCH --job-name=augur-collapsed-pair-term-scaling-probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-collapsed-pair-term-scaling-probe-%j.log
# Feasibility probe, audit row 11: adaptive pair-term time per body on a collapsed Plummer cloud (core 1) vs a reference cloud (core 4), N = 1e5, 3e5, 1e6, 3e6, seed 4738.
# Pre-registration: docs/debugging/director-feasibility-2026-11-07.md. Inference only, estimate under 0.5 GPU-h unless the collapsed 3e6 cell is slow (not timed; the earlier cell-list pair term took
# 89 s per tick at 1e6). Needs in $HOME/augur_nfr: kern/E_frozen.pt, scripts/scatter_field.py, scripts/pm_baseline_force_error.py and the new scripts/collapsed_pair_term_scaling_probe.py
# (upload it there). Per-cell results are saved after each cell and skipped on rerun: sbatch again to resume.
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
ln -sfn $HOME/augur_nfr/kern $HOME/augur_probe/kern; cd $HOME/augur_probe || { echo "MISSING $HOME/augur_probe"; exit 2; }
export PYTHONPATH=.
for f in kern/E_frozen.pt scripts/scatter_field.py scripts/scatter_scaling_ops.py scripts/pm_baseline_force_error.py scripts/collapsed_pair_term_scaling_probe.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/collapsed_pair_term_scaling_probe
mkdir -p $O
T0=$(date +%s)
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
dlbg python scripts/collapsed_pair_term_scaling_probe.py --ckpt kern/E_frozen.pt --out $O/probe.json --seed 4738 >> $O/probe.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $((DEADLINE-$(date +%s))) -le 300 ] && [ $RC -ne 0 ] && { echo "DEADLINE_STOP: unfinished, per-cell results saved"; exit 4; }
[ $RC -eq 0 ] || { echo "probe exit $RC"; tail -20 $O/probe.log; exit $RC; }
grep -q '^DONE' $O/probe.log && [ -s $O/probe.json ] || { echo "MISSING OUTPUT $O/probe.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
