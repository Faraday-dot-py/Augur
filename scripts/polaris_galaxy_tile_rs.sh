#!/bin/bash
#SBATCH --job-name=augur-galaxy-tile-rs
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=augur-galaxy-tile-rs-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-tile-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_tile2.tgz
K=$HOME/E_ms_kp_pot_v_g128.pt
K=$HOME/E_ms_kp_pot_v_g128.pt
R1B=file:results/far_kernels/r1b_mlp.pt+cache
A="--mode mesh --tile --kernel $R1B --n 1000000 --preset g1 --ckpt $K --steps 60 --diag-every 20 --ckpt-every 20 --rec-every 5 --nsnap 1000 --res 64"
python scripts/scatter_galaxy.py $A --tag rs_a
python scripts/scatter_galaxy.py $A --tag rs_b
python scripts/scatter_galaxy.py $A --tag rs_c --stop-at 40
python scripts/scatter_galaxy.py $A --tag rs_c --resume
python scripts/galaxy_resume_check.py rs_a rs_b
python scripts/galaxy_resume_check.py rs_a rs_c
echo "[$(date -Iseconds)] ALL DONE"
