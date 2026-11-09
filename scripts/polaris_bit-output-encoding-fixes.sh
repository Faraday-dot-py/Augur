#!/bin/bash
#SBATCH --job-name=augur-bit-output-encoding-fixes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --dependency=afterany:7191
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-bit-output-encoding-fixes-%j.log
# Bit-output cell MLP encoding fixes, one change per arm vs plain-bits and float twin, seeds 4738 then 4739/4740 for arms that beat plain-bits. Estimate 0.3-1.2 GPU-h (16 models per seed, 3000 steps each).
# Requeue-safe: results.json is rewritten after every model and finished (seed, arm, mode) keys are skipped on restart. Usage: sbatch this.sh (afterany:<slot 2 tail>)
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_bit_output_encoding
cd $W || exit 2
export PYTHONPATH=$W/scripts
for f in scripts/bit_force_smoke.py scripts/bit_output_encoding_fixes.py; do [ -e $f ] || { echo "MISSING $f"; exit 2; }; done
mkdir -p results/bit_output_encoding_fixes
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/bit_output_encoding_fixes.py --out results/bit_output_encoding_fixes/results.json &
PID=$!
wait $PID
rc=$?
[ $rc -eq 0 ] || { echo "FAILED rc=$rc"; exit $rc; }
grep -q "ALL DONE" augur-bit-output-encoding-fixes-${SLURM_JOB_ID}.log || { echo "MISSING completion marker"; exit 3; }
test -s results/bit_output_encoding_fixes/results.json || { echo "MISSING OUTPUT"; exit 3; }
echo EXIT 0
