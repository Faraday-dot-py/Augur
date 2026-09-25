#!/bin/bash
#SBATCH --job-name=fallback-b1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=fallback-b1-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results
R="python scripts/fallback_run.py"
$R --name cor_flyby_bias-2 --ic flyby --corrupt bias:-2
$R --name cor_flyby_bias-4 --ic flyby --corrupt bias:-4
$R --name cor_flyby_bias-6 --ic flyby --corrupt bias:-6
$R --name cor_flyby_healthy --ic flyby
