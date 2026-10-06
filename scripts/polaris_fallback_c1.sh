#!/bin/bash
#SBATCH --job-name=fallback-c1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=fallback-c1-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
R="python scripts/fallback_run.py"
$R --name fa_flyby_adapt --ic flyby --noise-ks 200 500 1000 3000 --true-every 20
$R --name fa_three_adapt --ic three --noise-ks 200 500 1000 3000 --true-every 20
$R --name fa_plummer_adapt --ic plummer --noise-ks 200 500 1000 3000 --true-every 20
$R --name fa_disk_adapt --ic disk --noise-ks 200 500 1000 3000 --true-every 20
$R --name fa_clumpy_adapt --ic clumpy --noise-ks 200 500 1000 3000 --true-every 20
