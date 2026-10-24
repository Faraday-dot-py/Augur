#!/bin/bash
#SBATCH --job-name=augur-fkg-ctx
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-fkg-ctx-%j.log
source $HOME/fkg_common.sh
for s in 4738 4739 4740; do
  python scripts/scatter_far_general.py cond --mode ctx --seed $s --steps 40000 > fkg_ctx_$s.log 2>&1 &
done
python scripts/scatter_far_general.py base --laws pow1,pow1.5,pow3,yuk8,yuk32,yuk128,lj,rep2,nr1b,scr32,nr2b --steps 30000 > fkg_base.log 2>&1 &
wait
tail -n 3 fkg_ctx_*.log
echo TRAIN DONE
