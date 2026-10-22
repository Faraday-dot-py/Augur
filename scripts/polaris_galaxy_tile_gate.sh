#!/bin/bash
#SBATCH --job-name=augur-galaxy-tile-gate
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-galaxy-tile-gate-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_tile1.tgz
K=$HOME/E_ms_kp_pot_v_g128.pt
R1B=file:results/far_kernels/r1b_mlp.pt+cache
python scripts/galaxy_tile_check.py --ckpt $K --n 20000
python scripts/galaxy_tile_check.py --ckpt $K --n 20000 --kernel $R1B
for P in g1 g2; do
  python scripts/scatter_galaxy.py --mode mesh --tile --kernel $R1B --n 100000 --steps 1000 --preset $P --ckpt $K --tag ${P}_tiled_100000
done
echo "[$(date -Iseconds)] ALL DONE"
