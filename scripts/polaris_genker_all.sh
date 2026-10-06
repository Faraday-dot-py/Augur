#!/bin/bash
#SBATCH --job-name=genker-all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --output=genker-all-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints

python scripts/genker_static.py --kernels inv_distance,yukawa30,learned --out results/genker_static_b.json
python scripts/genker_rollout.py --kernel lj --n 20000 --steps 600 --tag lj
python scripts/genker_rollout.py --kernel softgrav2 --n 20000 --steps 600 --tag softgrav2
python scripts/genker_rollout.py --kernel pow15 --n 20000 --steps 600 --tag pow15
python scripts/genker_rollout.py --kernel learned --n 20000 --steps 600 --providers exact,exact_pert,exact_analytic,adaptive_own,adaptive_transfer,geo_audit --tag learned
python scripts/genker_rollout.py --kernel inv_distance --n 20000 --steps 600 --tag inv_distance
python scripts/genker_rollout.py --kernel yukawa30 --n 20000 --steps 600 --tag yukawa30
