#!/bin/bash
#SBATCH --job-name=augur-law-recovery-and-pm-baseline-followups
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --nice=100
#SBATCH --output=augur-law-recovery-and-pm-baseline-followups-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in kern/E_frozen.pt ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/law_recovery_followup.py scripts/pm_baseline_force_error.py; do test -f $f || { echo MISSING $f; exit 2; }; done
python scripts/law_recovery_followup.py --out results/law_recovery_followup.json; rc1=$?
python scripts/pm_baseline_force_error.py --ckpt kern/E_frozen.pt --families plummer,two_clump,exp_disk,hernquist,three_clump --ns 300,1000 --scene-mult 4 --out results/pm_baseline_more_scenes_and_families.json; rc2=$?
python scripts/pm_baseline_force_error.py --ckpt kern/E_frozen.pt --families plummer,two_clump,exp_disk --ns 2000,4000 --density-scale --out results/pm_baseline_density_scaled.json; rc3=$?
echo EXIT $rc1 $rc2 $rc3
test -f results/law_recovery_followup.json -a -f results/pm_baseline_more_scenes_and_families.json -a -f results/pm_baseline_density_scaled.json || exit 3
exit $((rc1+rc2+rc3))
