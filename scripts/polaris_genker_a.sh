#!/bin/bash
#SBATCH --job-name=genker-a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=genker-a-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/genker_rollout.py --kernel learned --n 20000 --steps 10 --providers exact --diag-every 10 --snap-every 10 --tag time_learned
python scripts/genker_rollout.py --kernel lj --n 20000 --steps 10 --providers exact --diag-every 10 --snap-every 10 --tag time_lj
python scripts/genker_static.py --kernels lj,softgrav2,pow15 --out results/genker_static_a.json
