#!/bin/bash
#SBATCH --job-name=augur-galaxy-1b
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=augur-galaxy-1b-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_code1.tgz
for P in g1 g2; do
  T=${P}_mesh_1000000000
  RES=""; [ -d results/galaxy/$T/state ] && RES="--resume"
  python scripts/scatter_galaxy.py --mode mesh --n 1000000000 --steps 1000 --preset $P --ckpt $HOME/E_ms_kp_pot_v_g128.pt --tag $T $RES
done
echo "[$(date -Iseconds)] ALL DONE"
