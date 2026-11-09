#!/bin/bash
#SBATCH --job-name=augur-collapsed-pair-term-fallback-chase
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
#SBATCH --dependency=afterany:7183
#SBATCH --output=augur-collapsed-pair-term-fallback-chase-%j.log
# Curiosity diagnosis: fallback counts, over-cap counts and per-stage times of the adaptive pair term on a collapsed Plummer cloud, N = 1e6, 1.5e6, 2e6, 3e6, seed 4738.
# Pre-registration: docs/debugging/experiment-log.md (2026-11-08 collapsed-pair-term-fallback-chase). Inference only, estimate under 40 min (not timed).
# Needs in $HOME/augur_probe: kern (symlink to augur_nfr/kern), scripts/scatter_field.py (instrumented), scripts/pm_baseline_force_error.py, scripts/collapsed_pair_term_fallback_chase.py.
# Per-cell results are saved after each cell and skipped on rerun: sbatch again to resume.
set -uo pipefail
export PYTHONUNBUFFERED=1
ln -sfn $HOME/augur_nfr/kern $HOME/augur_probe/kern; cd $HOME/augur_probe || { echo "MISSING $HOME/augur_probe"; exit 2; }
export PYTHONPATH=.
for f in kern/E_frozen.pt scripts/scatter_field.py scripts/pm_baseline_force_error.py scripts/collapsed_pair_term_fallback_chase.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q adapt_profile scripts/scatter_field.py || { echo "STALE scatter_field.py: no adapt_profile"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/collapsed_pair_term_fallback_chase
mkdir -p $O
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/collapsed_pair_term_fallback_chase.py --ckpt kern/E_frozen.pt --out $O/chase.json --seed 4738 >> $O/chase.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "chase exit $RC"; tail -20 $O/chase.log; exit $RC; }
grep -q '^DONE' $O/chase.log && [ -s $O/chase.json ] || { echo "MISSING OUTPUT $O/chase.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
