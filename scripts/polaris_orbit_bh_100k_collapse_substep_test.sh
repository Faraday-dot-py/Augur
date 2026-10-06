#!/bin/bash
#SBATCH --job-name=augur-orbit-bh-100k-collapse-substep-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=augur-orbit-bh-100k-collapse-substep-test-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
for S in 1 2 4; do
    echo "== substeps $S"
    python scripts/orbit_bh.py --mode blackhole --n 100000 --steps 60 --substeps $S --record 100000 --diag 2 --ratio 0.1 --c 200 --relativistic --sigma 10 --force exact --kernel learned_rel --tag bh_100k_collapse_sub$S
done
