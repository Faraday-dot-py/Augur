#!/bin/bash
#SBATCH --job-name=augur-sae-curiosity-mixed-s4738-slow-l0-control
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --output=augur-sae-curiosity-mixed-s4738-slow-l0-control-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
test -f ckpt/main_s4738.pt || { echo MISSING ckpt; exit 2; }
test -f scripts/sae_interp.py || { echo MISSING script; exit 2; }
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --out results/sae_interp_s4738_slowcontrol.json
rc=$?
test -f results/sae_interp_s4738_slowcontrol.json || { echo MISSING OUTPUT; exit 3; }
echo EXIT $rc
exit $rc
