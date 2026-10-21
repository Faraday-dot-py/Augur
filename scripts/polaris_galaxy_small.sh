#!/bin/bash
#SBATCH --job-name=augur-galaxy-small
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-galaxy-small-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_code1.tgz
K=$HOME/E_ms_kp_pot_v_g128.pt
for P in g1 g2; do
  for N in 100000 1000000 10000000; do
    python scripts/scatter_galaxy.py --mode mesh --n $N --steps 1000 --preset $P --ckpt $K --tag ${P}_mesh_$N
  done
  T=${P}_truth_100000
  RES=""; [ -d results/galaxy/$T/state ] && RES="--resume"
  python scripts/scatter_galaxy.py --mode truth --n 100000 --steps 1000 --preset $P --tag $T $RES
done
echo "[$(date -Iseconds)] ALL DONE"
