#!/bin/bash
#SBATCH --job-name=augur-scatter-tile2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=augur-scatter-tile2-%j.log

# Minutes-scale jobs (no resume/ckpt needed). Commands to run are in $HOME/augur_tile2/job_cmds.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_tile2"
export PYTHONPATH=.
mkdir -p results
nvidia-smi --query-gpu=name,memory.used --format=csv
bash job_cmds.sh
