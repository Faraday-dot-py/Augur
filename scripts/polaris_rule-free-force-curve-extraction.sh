#!/bin/bash
#SBATCH --job-name=augur-rule-free-force-curve-extraction
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --dependency=afterany:7187
#SBATCH --nice=100
#SBATCH --signal=B:TERM@60
#SBATCH --output=augur-rule-free-force-curve-extraction-%j.log
# Curve extraction for the rule-free force recovery experiment; per-cell results are saved and skipped on rerun (sbatch again to resume). Estimate ~40 min, ~0.7 GPU-h
# (scaled from job 4832: 504 distance cells in 5.3 min; this job has 3568 cells). Needs in $HOME/augur_nfr: ckpt/main_s4738.pt, ckpt/main_s4739.pt, scripts/rule_free_force_curves.py.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr || { echo "MISSING $HOME/augur_nfr"; exit 2; }
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/rule_free_force_curves.py scripts/law_recovery_followup.py scripts/law_recovery_probe.py; do test -f $f || { echo MISSING $f; exit 2; }; done
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/rule_free_force_curves.py --out results/rule_free_force_curves &
PID=$!
wait $PID
rc=$?
n=$(ls results/rule_free_force_curves/*.npz 2>/dev/null | wc -l)
[ "$n" -eq 16 ] || { echo "INCOMPLETE OUTPUT $n of 16"; exit 3; }
echo EXIT $rc
exit $rc
