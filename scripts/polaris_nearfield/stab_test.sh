#!/bin/bash
#SBATCH --job-name=augur-nfr-stab
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-nfr-stab-%j.log
source $HOME/nfr_common.sh
rm -rf ckpt_stab
python scripts/train_scatter_multi.py --tag stab --iters 8000 --per-shape 1500 --ckpt-every 500 --log-every 250 --ckpt-dir ckpt_stab
echo STAB DONE
