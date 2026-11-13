#!/bin/bash
#SBATCH --job-name=augur-scatter-field-three-part-exact-swap-drift-attribution
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SUPERVISOR_REPLACES_WITH_SLOT_TAIL_JOB_ID
#SBATCH --output=augur-scatter-field-three-part-exact-swap-drift-attribution-%j.log
# Evaluation only, no training. Pre-registration: docs/debugging/architecture-push-prereg/scatter-field-three-part-exact-swap-drift-attribution.md.
# Estimate 0.2-0.5 GPU-h (CPU smoke 10-20 s per group at 2 scenes x 25 ticks for exp-e-frozen; GPU rate not measured, untested). Finished groups are skipped on rerun: sbatch again to resume.
# Needs $HOME/augur-scatter-field-three-part-exact-swap-drift-attribution-code.tgz, $HOME/augur_nfr/kern/E_frozen.pt and $HOME/augur-single-law-network-sizing/ckpt/width8.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
CODE=$HOME/augur-scatter-field-three-part-exact-swap-drift-attribution-code.tgz
E=$HOME/augur_nfr/kern/E_frozen.pt
C=$HOME/augur-single-law-network-sizing/ckpt/width8.pt
for f in $CODE $E $C; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
tar -tzf $CODE | grep -x scripts/scatter_three_part_exact_swap_drift_attribution.py > /dev/null || { echo "TARBALL LACKS RUN SCRIPT"; exit 2; }
W=$HOME/augur-scatter-field-three-part-exact-swap-drift-attribution
mkdir -p $W/results || exit 2
SUM=$(md5sum $CODE | cut -d' ' -f1)
[ -e $W/.extracted-$SUM ] || { tar xzf $CODE -C $W && touch $W/.extracted-$SUM; } || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=$W
for f in scripts/scatter_field.py scripts/scatter_multi.py scripts/single_law_network_sizing.py scripts/scatter_error_growth.py scripts/scatter_three_part_exact_swap_drift_attribution.py scripts/scatter_three_part_exact_swap_drift_attribution_analyze.py; do
  [ -s $f ] || { echo "MISSING $f in tarball"; exit 2; }
done
md5sum scripts/scatter_three_part_exact_swap_drift_attribution.py scripts/scatter_three_part_exact_swap_drift_attribution_analyze.py $E $C
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/scatter-field-three-part-exact-swap-drift-attribution
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for m in exp-e-frozen:$E width8-single-law-control:$C; do
  python3 scripts/scatter_three_part_exact_swap_drift_attribution.py --model ${m%%:*} --ckpt ${m#*:} --out-dir $O >> three-part-swap.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "${m%%:*} exit $RC"; tail -20 three-part-swap.log; exit $RC; }
  [ -e $O/${m%%:*}.done ] || { echo "MISSING $O/${m%%:*}.done"; exit 1; }
done
[ $(ls $O/*.npz | wc -l) -eq 24 ] || { echo "EXPECTED 24 group files, found $(ls $O/*.npz | wc -l)"; exit 1; }
python3 scripts/scatter_three_part_exact_swap_drift_attribution_analyze.py --in-dir $O || exit 1
[ -s $O/attribution.json ] || { echo "MISSING attribution.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
