#!/bin/bash
#SBATCH --job-name=augur-interior-force-offset-corrected-quantity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:7242
#SBATCH --output=augur-interior-force-offset-corrected-quantity-%j.log
# Audit row 10 rerun on the quantity behind the 1.55x figure (grid |a| vs exact point sum at cell centres, black-hole scene). Pre-registration: docs/debugging/experiment-log.md 2026-11-08 interior-force-offset-corrected-quantity.
# Seed 4738, inference only, three arms in one job, estimate under 15 min (not timed). Per-arm results are saved and skipped on rerun: sbatch again to resume.
# Needs $HOME/augur-edgefree/{scripts,model} (scatter_bh_1b.py, scatter_field.py with kpad) plus the uploaded scripts/interior_force_offset_corrected_quantity.py and $HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
D=$HOME/augur-edgefree
cd $D || { echo "MISSING $D"; exit 2; }
export PYTHONPATH=.
CK=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
for f in $CK scripts/scatter_field.py scripts/scatter_bh_1b.py scripts/scatter_bh.py scripts/scatter_scaling_ops.py scripts/train_scatter_field.py scripts/interior_force_offset_corrected_quantity.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q kpad scripts/scatter_field.py || { echo "STALE scatter_field.py: no kpad"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/interior_force_offset_corrected_quantity
mkdir -p $O
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/interior_force_offset_corrected_quantity.py --ckpt $CK --out $O/result.json --seed 4738 >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "run exit $RC"; tail -20 $O/run.log; exit $RC; }
grep -q '^DONE' $O/run.log && [ -s $O/result.json ] || { echo "MISSING OUTPUT $O/result.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
