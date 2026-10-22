#!/bin/bash
#SBATCH --job-name=augur-galaxy-tile-1b
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=augur-galaxy-tile-1b-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-tile-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_tile1.tgz
K=$HOME/E_ms_kp_pot_v_g128.pt
R1B=file:results/far_kernels/r1b_mlp.pt+cache
A="--mode mesh --tile --kernel $R1B --n 1000000 --preset g1 --ckpt $K --diag-every 20 --ckpt-every 20 --rec-every 5 --nsnap 1000 --res 64"
python scripts/scatter_galaxy.py $A --steps 40 --tag rs_straight
python scripts/scatter_galaxy.py $A --steps 20 --tag rs_resume
python scripts/scatter_galaxy.py $A --steps 40 --tag rs_resume --resume
python scripts/galaxy_resume_check.py rs_straight rs_resume
T=g1_tiled_1000000000
RES=""; [ -d results/galaxy/$T/state ] && RES="--resume"
python scripts/scatter_galaxy.py --mode mesh --tile --kernel $R1B --n 1000000000 --steps 1000 --preset g1 --ckpt $K --tag $T $RES
echo "[$(date -Iseconds)] ALL DONE"
