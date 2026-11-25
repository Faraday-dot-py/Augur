#!/bin/bash
#SBATCH --job-name=augur-fkg-smoke
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=augur-fkg-smoke-%j.log
source $HOME/fkg_common.sh
mkdir -p kern/gensmoke
python scripts/scatter_far_general.py base --laws pow1.5,lj --steps 1000 --outdir kern/gensmoke
python scripts/scatter_far_general.py cond --mode theta --steps 300 --outdir kern/gensmoke
python scripts/scatter_far_general.py cond --mode ctx --steps 300 --outdir kern/gensmoke
echo SMOKE DONE
