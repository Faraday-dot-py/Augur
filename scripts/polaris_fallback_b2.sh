#!/bin/bash
#SBATCH --job-name=fallback-b2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=fallback-b2-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
R="python scripts/fallback_run.py"
$R --name cor_flyby_noise --ic flyby --corrupt noise
$R --name cor_flyby_const-10 --ic flyby --corrupt const:-10
$R --name cor_flyby_const-4 --ic flyby --corrupt const:-4
$R --name cor_flyby_noiseasym --ic flyby --corrupt noise_asym --steps 200
$R --name cor_flyby_geo --ic flyby --mode geo_audit
