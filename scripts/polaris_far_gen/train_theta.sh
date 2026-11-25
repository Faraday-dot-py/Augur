#!/bin/bash
#SBATCH --job-name=augur-fkg-theta
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-fkg-theta-%j.log
source $HOME/fkg_common.sh
for s in 4738 4739 4740; do
  python scripts/scatter_far_general.py cond --mode theta --seed $s --steps 40000 > fkg_theta_$s.log 2>&1 &
done

wait
tail -n 3 fkg_theta_*.log
echo TRAIN DONE
