#!/bin/bash
#SBATCH --job-name=augur-gpu-burn-filler
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=augur-gpu-burn-filler-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
test -f scripts/gpu_burn_check.py || { echo MISSING scripts/gpu_burn_check.py; exit 2; }
DEADLINE=1794070800
REMAIN=$((DEADLINE - $(date +%s)))
if [ $REMAIN -le 300 ]; then echo PAST_DEADLINE; exit 3; fi
mkdir -p results/gpu_burn_check
echo HOST $(hostname) JOB $SLURM_JOB_ID CVD ${CUDA_VISIBLE_DEVICES:-unset} REMAIN $REMAIN
nvidia-smi -L
python scripts/gpu_burn_check.py --seconds $REMAIN --seed 4738 --out results/gpu_burn_check/filler_${SLURM_JOB_ID}.json
rc=$?
echo EXIT $rc
exit $rc
