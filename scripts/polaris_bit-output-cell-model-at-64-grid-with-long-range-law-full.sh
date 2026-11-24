#!/bin/bash
#SBATCH --job-name=augur-bit-output-64-grid-long-range-full
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --dependency=afterany:SLOT_TAIL_PLACEHOLDER
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-bit-output-64-grid-long-range-full-%j.log
# NOT YET AUTHORISED. The supervisor submits this only if the stage 1 job gave PASS on both eval sets (held-out and fresh) for training seed 4738. Training seeds 4739 and 4740, same settings, same out dir.
# Estimate 0.5 GPU-h (spec cap 1.5). Requeue-safe as in stage 1.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_bit_output_64
cd $W || exit 2
export PYTHONPATH=$W/scripts
for f in scripts/bit_force_smoke.py scripts/bit_output_encoding_fixes.py scripts/bit_output_64_grid_long_range.py results/bit_output_64_grid/results.json; do [ -e $f ] || { echo "MISSING $f"; exit 2; }; done
npass=$(grep -c '"label": "PASS"' results/bit_output_64_grid/results.json)
nlab=$(grep -c '"label":' results/bit_output_64_grid/results.json)
[ "$npass" -ge 2 ] && [ "$npass" -eq "$nlab" ] || { echo "STAGE 1 NOT PASS (labels $nlab, pass $npass): refusing to run"; exit 2; }
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for extra in "" "--fresh"; do
  python scripts/bit_output_64_grid_long_range.py --out-dir results/bit_output_64_grid --seeds 4739,4740 --resume $extra &
  PID=$!
  wait $PID
  rc=$?
  [ $rc -eq 0 ] || { echo "FAILED rc=$rc extra=$extra"; exit $rc; }
done
test -s results/bit_output_64_grid/results.json || { echo "MISSING OUTPUT"; exit 3; }
echo EXIT 0
