#!/bin/bash
#SBATCH --job-name=augur-width-32-loss-spike-instrumented-rerun
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --dependency=afterany:7475
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-width-32-loss-spike-instrumented-rerun-%j.log
# Width 32 seed 4740 to it 3300 then seed 4738 to it 3000, recipe unchanged (md5 of the base script 63b98665... checked below), logging only via --trace-file; full-state ckpt kept at it 2500.
# Pre-registration: docs/debugging/experiment-log.md 2026-11-10 width-32-loss-spike-instrumented-rerun. Estimate 0.3 GPU-h (untimed, 0.15 s/it from job 7442 plus trace overhead).
# Requeue-safe: ckpt every 500 its, --resume implicit, finished seeds skipped via DONE files; sbatch again (afterany:<slot 2 tail>) to resume.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur-single-law-network-sizing
U=$HOME/augur-single-law-upload
V=$HOME/augur-width32-spike-instrumented-upload
[ -d $W/scripts ] && [ -d $W/model ] || { echo "MISSING base $W"; exit 2; }
for f in scatter_field.py scatter_multi.py single_law_network_sizing.py; do
  [ -e $U/$f ] || { echo "MISSING upload $f"; exit 2; }
  cp $U/$f $W/scripts/$f
done
[ -e $V/single_law_network_sizing_instrumented.py ] || { echo "MISSING instrumented script"; exit 2; }
cp $V/single_law_network_sizing_instrumented.py $W/scripts/single_law_network_sizing_instrumented.py
cd $W || exit 2
export PYTHONPATH=$W
grep -q kpad scripts/scatter_field.py || { echo "STALE scatter_field.py"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/single_law_network_sizing.py scripts/single_law_network_sizing_instrumented.py
[ "$(md5sum < scripts/single_law_network_sizing.py | cut -d' ' -f1)" = 63b9866516b20f80a3bca1e5b0232eff ] || { echo "BASE RECIPE MD5 MISMATCH"; exit 2; }
[ "$(md5sum < scripts/scatter_field.py | cut -d' ' -f1)" = 47b903e757cf1221dd71d10f47b7377a ] || { echo "scatter_field MD5 MISMATCH"; exit 2; }
[ "$(md5sum < scripts/scatter_multi.py | cut -d' ' -f1)" = 7b96aec0dc1536da94126dd7a8699953 ] || { echo "scatter_multi MD5 MISMATCH"; exit 2; }
[ "$(md5sum < scripts/single_law_network_sizing_instrumented.py | cut -d' ' -f1)" = 3930f95c01e89c2f91aaa697c54c1aed ] || { echo "INSTRUMENTED MD5 MISMATCH"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/width32_loss_spike_instrumented_rerun
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for SPEC in 4740:3300 4738:3000; do
  S=${SPEC%%:*}
  STOP=${SPEC##*:}
  mkdir -p $O/seed$S ckpt_width32_spike_instrumented/seed$S
  [ -e $O/seed$S/DONE ] && { echo "seed $S already done"; continue; }
  python scripts/single_law_network_sizing_instrumented.py --widths 32 --seed $S --out-dir $O/seed$S --ckpt-dir ckpt_width32_spike_instrumented/seed$S --trace-file $O/seed$S/trace.jsonl --stop-at $STOP --keep-ckpt-at 2500 >> $O/seed$S/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "seed $S exit $RC"; tail -20 $O/seed$S/run.log; exit $RC; }
  grep -q "reached --stop-at $STOP" $O/seed$S/run.log && [ -s $O/seed$S/trace.jsonl ] && [ -s ckpt_width32_spike_instrumented/seed$S/width32_it2500.pt ] || { echo "MISSING OUTPUT seed $S"; exit 1; }
  touch $O/seed$S/DONE
done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
