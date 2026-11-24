#!/bin/bash
#SBATCH --job-name=augur-bit-output-64-grid-long-range
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --dependency=afterany:SLOT_TAIL_PLACEHOLDER
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-bit-output-64-grid-long-range-%j.log
# Stage 1: plain bits (soft decode) vs float twin, 64x64 grid, softened inverse-square, training seed 4738, held-out eval seeds 9100/9200/9300 then fresh 12000/12100/12200 on the same trained models. Estimate 0.25 GPU-h.
# Requeue-safe: checkpoints every 250 steps, finished models and eval keys are skipped on restart. Supervisor replaces SLOT_TAIL_PLACEHOLDER with the slot tail job id before sbatch.
set -uo pipefail
export PYTHONUNBUFFERED=1
TGZ=${TGZ:-$HOME/augur-architecture-push-single-law-readability-code.tgz}
W=$HOME/augur_bit_output_64
test -f $TGZ || { echo MISSING $TGZ; exit 2; }
mkdir -p $W && tar xzf $TGZ -C $W || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=$W/scripts
for f in scripts/bit_force_smoke.py scripts/bit_output_encoding_fixes.py scripts/bit_output_64_grid_long_range.py; do [ -e $f ] || { echo "MISSING $f"; exit 2; }; done
mkdir -p results/bit_output_64_grid
nvidia-smi --query-gpu=name,memory.total --format=csv
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for extra in "" "--fresh"; do
  python scripts/bit_output_64_grid_long_range.py --out-dir results/bit_output_64_grid --seeds 4738 --resume $extra &
  PID=$!
  wait $PID
  rc=$?
  [ $rc -eq 0 ] || { echo "FAILED rc=$rc extra=$extra"; exit $rc; }
done
test -s results/bit_output_64_grid/results.json || { echo "MISSING OUTPUT"; exit 3; }
grep -q '"4738/held-out"' results/bit_output_64_grid/results.json && grep -q '"4738/fresh"' results/bit_output_64_grid/results.json || { echo "MISSING VERDICTS"; exit 3; }
echo EXIT 0
