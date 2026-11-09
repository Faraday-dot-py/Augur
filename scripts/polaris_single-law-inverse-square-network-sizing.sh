#!/bin/bash
#SBATCH --job-name=augur-single-law-inverse-square-network-sizing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:7249
#SBATCH --output=augur-single-law-inverse-square-network-sizing-%j.log
# UNet width 4/8/16/32/64 on inverse-square gravity only, seed 4738, held-out seeds 4739/4740. Pre-registration: docs/debugging/experiment-log.md 2026-11-08 single-law-inverse-square-network-sizing.
# Estimate 3.5-4 h (untimed, from 0.315 s/it of jobs 3898/3899). Requeue-safe: per-width ckpt every 500 its, finished widths skipped; sbatch again (afterany:<slot 2 tail>) to resume.
# Needs $HOME/augur_nfr (scripts, model) as a base; local scatter_field.py, scatter_multi.py and single_law_network_sizing.py are uploaded to $HOME/augur-single-law-upload and overlaid.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur-single-law-network-sizing
U=$HOME/augur-single-law-upload
if [ ! -d $W/scripts ]; then
  mkdir -p $W
  cp -r $HOME/augur_nfr/scripts $HOME/augur_nfr/model $W/ || { echo "COPY FAILED"; exit 2; }
fi
for f in scatter_field.py scatter_multi.py single_law_network_sizing.py; do
  [ -e $U/$f ] || { echo "MISSING upload $f"; exit 2; }
  cp $U/$f $W/scripts/$f
done
cd $W || exit 2
export PYTHONPATH=$W
grep -q kpad scripts/scatter_field.py || { echo "STALE scatter_field.py"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/single_law_network_sizing.py
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/single_law_network_sizing
mkdir -p $O ckpt
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/single_law_network_sizing.py --out-dir $O --ckpt-dir ckpt >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "run exit $RC"; tail -20 $O/run.log; exit $RC; }
grep -q '^ALL DONE' $O/run.log && [ -s $O/verdict.json ] || { echo "MISSING OUTPUT $O/verdict.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
