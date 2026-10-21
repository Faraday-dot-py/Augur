#!/bin/bash
#SBATCH --job-name=augur-esc-smoke
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=augur-esc-smoke-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_escape"
cd "$HOME/augur_escape"
tar xzf "$HOME/escape_code2.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
python scripts/escape_roll.py --ckpt $CK --n 3000 --seed 4738 --steps 120 --arms S0,S1,S1nm,S1g,T4,T8,T8w --truth --probe-ticks 60,120 --snap-every 10 --out results/escape/smoke
python scripts/escape_analyze.py results/escape/smoke
python scripts/escape_roll.py --ckpt $CK --n 3000 --nt 200 --seed 4738 --steps 60 --arms S1,T4 --truth --out results/escape/smoke_tr
python scripts/escape_analyze.py results/escape/smoke_tr
echo DONE
