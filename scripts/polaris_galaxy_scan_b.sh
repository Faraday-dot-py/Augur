#!/bin/bash
#SBATCH --job-name=augur-galaxy-scan-b
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=augur-galaxy-scan-b-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_code1.tgz
for P in g1 g2; do
  python scripts/scatter_galaxy.py --mode truth --n 100000 --steps 300 --preset $P --tag scan_truth_${P}_100000
done
echo "[$(date -Iseconds)] ALL DONE"
