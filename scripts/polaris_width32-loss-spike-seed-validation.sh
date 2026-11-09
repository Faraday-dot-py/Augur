#!/bin/bash
#SBATCH --job-name=augur-width32-loss-spike-seed-validation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:7300
#SBATCH --output=augur-width32-loss-spike-seed-validation-%j.log
# UNet width 32 only, training seeds 4739/4740/4741 sequentially, same recipe as the width-32 arm of job 7256. Pre-registration: docs/debugging/experiment-log.md 2026-11-09 width-32-loss-spike-seed-validation.
# Estimate 2.2 h (untimed, from 2461 s per width-32 run in 7256). Requeue-safe: per-seed ckpt every 500 its, finished seeds skipped; sbatch again (afterany:<slot 2 tail>) to resume.
# Reuses $HOME/augur-single-law-network-sizing (scripts, model base) and the 7256 upload dir.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur-single-law-network-sizing
U=$HOME/augur-single-law-upload
[ -d $W/scripts ] && [ -d $W/model ] || { echo "MISSING base $W"; exit 2; }
for f in scatter_field.py scatter_multi.py single_law_network_sizing.py; do
  [ -e $U/$f ] || { echo "MISSING upload $f"; exit 2; }
  cp $U/$f $W/scripts/$f
done
cd $W || exit 2
export PYTHONPATH=$W
grep -q kpad scripts/scatter_field.py || { echo "STALE scatter_field.py"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/single_law_network_sizing.py
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/width32_loss_spike_validation
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for S in 4739 4740 4741; do
  mkdir -p $O/seed$S ckpt_width32_spike/seed$S
  python scripts/single_law_network_sizing.py --widths 32 --seed $S --out-dir $O/seed$S --ckpt-dir ckpt_width32_spike/seed$S >> $O/seed$S/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "seed $S exit $RC"; tail -20 $O/seed$S/run.log; exit $RC; }
  grep -q '^ALL DONE' $O/seed$S/run.log && [ -s $O/seed$S/width32.json ] || { echo "MISSING OUTPUT seed $S"; exit 1; }
  rm -f $O/seed$S/verdict.json
done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
