#!/bin/bash
#SBATCH --job-name=augur-tier0-tiled-closed-loop-tests-scorer-check-tolerance-fix-on-merge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --output=/home/adamwebb/augur-tier0-tiled-closed-loop-tests-scorer-check-tolerance-fix-on-merge-%j.log
# Single stage verification of scripts/tier0_tiled_closed_loop_tests.py on the scratch-merge branch with the fresh-seed scorer-check tolerance fix (code tarball merge_scorer_check_tolerance_fix_code.tgz). Tests only, under 0.1 GPU-h.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_merge_scorer_fix_tests
rm -rf $W && mkdir -p $W && cd $W && tar xzf $HOME/merge_scorer_check_tolerance_fix_code.tgz
export PYTHONPATH=$W
nvidia-smi -L
python -c "import torch; assert torch.cuda.is_available(), 'no cuda'; print(torch.__version__, torch.cuda.get_device_name(0))" || { echo "RERUN_FAILED: no cuda"; exit 2; }
python scripts/tier0_tiled_closed_loop_tests.py --device cuda
rc=$?
echo "=== STAGE tier0-tiled-closed-loop-tests rc=$rc"
exit $rc
