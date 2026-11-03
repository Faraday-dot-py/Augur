#!/bin/bash
#SBATCH --job-name=augur-pm-baseline-force-error
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --nice=100
#SBATCH --output=augur-pm-baseline-force-error-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
test -f kern/E_frozen.pt || { echo MISSING E_frozen; exit 2; }
test -f scripts/pm_baseline_force_error.py || { echo MISSING script; exit 2; }
python scripts/pm_baseline_force_error.py --ckpt kern/E_frozen.pt --out results/pm_baseline_force_error.json
rc=$?
test -f results/pm_baseline_force_error.json || { echo MISSING OUTPUT; exit 3; }
echo EXIT $rc
exit $rc
