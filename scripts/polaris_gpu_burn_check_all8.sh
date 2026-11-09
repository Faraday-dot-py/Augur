#!/bin/bash
#SBATCH --job-name=augur-gpu-burn-check-per-gpu
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=augur-gpu-burn-check-per-gpu-%A_%a.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
test -f scripts/gpu_burn_check.py || { echo MISSING scripts/gpu_burn_check.py; exit 2; }
mkdir -p results/gpu_burn_check
echo HOST $(hostname) TASK $SLURM_ARRAY_TASK_ID CVD ${CUDA_VISIBLE_DEVICES:-unset}
nvidia-smi -L
python scripts/gpu_burn_check.py --seconds ${BURN_SECONDS:-1200} --seed 4738 --out results/gpu_burn_check/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json
rc=$?
echo EXIT $rc
nvidia-smi --query-gpu=uuid,ecc.errors.uncorrected.volatile.total,clocks_throttle_reasons.active --format=csv
exit $rc
