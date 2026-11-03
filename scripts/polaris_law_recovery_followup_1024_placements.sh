#!/bin/bash
#SBATCH --job-name=augur-law-recovery-followup-1024-placements
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --dependency=afterany:4828
#SBATCH --output=augur-law-recovery-followup-1024-placements-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/law_recovery_followup.py; do test -f $f || { echo MISSING $f; exit 2; }; done
python scripts/law_recovery_followup.py --n 1024 --out results/law_recovery_followup_1024_placements.json
rc=$?
test -f results/law_recovery_followup_1024_placements.json || { echo MISSING OUTPUT; exit 3; }
echo EXIT $rc
exit $rc
