#!/bin/bash
#SBATCH --job-name=interp-all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=interp-all-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints
date
python scripts/interp_train.py --kernels analytic,inv_distance,yukawa30,plw0.5,plw1.5,plw3 --out results/interp_train.json
date
python scripts/interp_stage1.py --out results/interp_stage1.json
date
python scripts/interp_maps.py --out results/interp_maps.json
date
python scripts/interp_decode.py --out results/interp_decode.json
date
