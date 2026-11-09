#!/bin/bash
#SBATCH --job-name=augur-mixed-law-checkpoint-probes
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --dependency=afterany:7269
#SBATCH --output=augur-mixed-law-checkpoint-probes-%j.log
# Dec3 activation by law and amplitude (curiosity, no verdict), then shift-equivariance probe (pass line: median > 2% on >= 3 of 6 laws). Inference only, 1 GPU.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/mixed_law_dec3_activation_by_law_and_amplitude.py scripts/mixed_law_shift_equivariance_probe.py; do test -f $f || { echo MISSING $f; exit 2; }; done
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/mixed_law_dec3_activation_by_law_and_amplitude.py --out results/mixed_law_dec3_activation --device cuda
rc1=$?
echo dec3_activation exit $rc1
python scripts/mixed_law_shift_equivariance_probe.py --out results/mixed_law_shift_equivariance_probe --device cuda
rc2=$?
echo shift_equivariance exit $rc2
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
