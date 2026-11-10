#!/bin/bash
#SBATCH --job-name=augur-width-32-loss-spike-eight-repeats-fixed-seed-4740
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-width-32-loss-spike-eight-repeats-fixed-seed-4740-%j.log
# Width 32 seed 4740 run 8 times to it 3300, recipe unchanged (base md5 63b98665... via the instrumented copy 3930f95c...), trace logging on; each repeat in its own dir.
# Pre-registration: docs/debugging/experiment-log.md 2026-11-10 width-32-loss-spike-eight-repeats-fixed-seed-4740. Estimate about 10.5 min per repeat (0.17 s/it from job 7777 x 3300 its plus start-up), about 1.4 GPU-h total, limit 2.5 h.
# Requeue-safe: ckpt every 500 its, --resume implicit, finished repeats skipped via DONE files; sbatch again with --dependency=afterany:<slot tail> to resume (a repeat spanning a resume is flagged and scored from the trace). Submit with a dependency so no more than 2 GPU jobs run at once.
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
O=results/width32_loss_spike_eight_repeats_fixed_seed_4740
STOP=3300
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for R in 1 2 3 4 5 6 7 8; do
  mkdir -p $O/repeat$R ckpt_width32_spike_repeats/repeat$R
  [ -e $O/repeat$R/DONE ] && { echo "repeat $R already done"; continue; }
  python scripts/single_law_network_sizing_instrumented.py --widths 32 --seed 4740 --out-dir $O/repeat$R --ckpt-dir ckpt_width32_spike_repeats/repeat$R --trace-file $O/repeat$R/trace.jsonl --stop-at $STOP --keep-ckpt-at 2500 >> $O/repeat$R/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "repeat $R exit $RC"; tail -20 $O/repeat$R/run.log; exit $RC; }
  grep -q "reached --stop-at $STOP" $O/repeat$R/run.log && [ -s $O/repeat$R/trace.jsonl ] && [ -s ckpt_width32_spike_repeats/repeat$R/width32_it2500.pt ] || { echo "MISSING OUTPUT repeat $R"; exit 1; }
  touch $O/repeat$R/DONE
done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
