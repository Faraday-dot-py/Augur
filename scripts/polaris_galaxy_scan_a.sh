#!/bin/bash
#SBATCH --job-name=augur-galaxy-scan-a
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=augur-galaxy-scan-a-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
D=$HOME/augur-galaxy-$SLURM_JOB_ID; mkdir -p $D/results/galaxy && cd $D
tar xzf $HOME/galaxy_code1.tgz
K=$HOME/E_ms_kp_pot_v_g128.pt
for P in g1 g2; do
  python scripts/galaxy_ic_check.py --preset $P --out results/galaxy/ic_check_$P.json
  python scripts/scatter_galaxy.py --mode mesh --n 100000 --steps 600 --preset $P --ckpt $K --tag scan_mesh_${P}_100000
done
echo "[$(date -Iseconds)] ALL DONE"
