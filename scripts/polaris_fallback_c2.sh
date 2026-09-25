#!/bin/bash
#SBATCH --job-name=fallback-c2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=fallback-c2-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results
R="python scripts/fallback_run.py"
$R --name fa_flyby_geo --ic flyby --mode geo_audit --true-every 20
$R --name fa_three_geo --ic three --mode geo_audit --true-every 20
$R --name fa_plummer_geo --ic plummer --mode geo_audit --true-every 20
$R --name fa_disk_geo --ic disk --mode geo_audit --true-every 20
$R --name fa_clumpy_geo --ic clumpy --mode geo_audit --true-every 20
