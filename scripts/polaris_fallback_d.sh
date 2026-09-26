#!/bin/bash
#SBATCH --job-name=fallback-d
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=fallback-d-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results
R="python scripts/fallback_run.py"
$R --name coll_adaptive --ic collapse --mode adaptive --steps 3000 --true-every 10 --diag-every 150
$R --name coll_geo_audit --ic collapse --mode geo_audit --steps 3000 --true-every 10 --diag-every 150
$R --name coll_est --ic collapse --mode est --steps 3000 --true-every 10 --diag-every 150
$R --name fa_clumpy_adapt --ic clumpy --noise-ks 200 500 1000 3000 --true-every 20
