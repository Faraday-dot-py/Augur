#!/bin/bash
#SBATCH --job-name=augur-nfr-smoke
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=augur-nfr-smoke-%j.log
source $HOME/nfr_common.sh
python -m pytest tests/test_scatter_multi.py -x -q 2>&1 | tail -20
rm -rf ckpt_smoke
python scripts/train_scatter_multi.py --tag smoke --iters 40 --per-shape 48 --ckpt-every 10 --snap-every 20 --log-every 10 --ckpt-dir ckpt_smoke --k-end 8
echo SMOKE DONE
