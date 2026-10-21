#!/bin/bash
#SBATCH --job-name=augur-esc-E
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=augur-esc-E-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_escape"
cd "$HOME/augur_escape"
tar xzf "$HOME/escape_code5.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
for S in 4738 9000 12000; do
python scripts/escape_roll.py --ckpt $CK --n 100000 --seed $S --steps 1000 --arms S1,T8,S1in,T8in,S1un,T4,T8w --resume --ckpt-every 250 --snap-every 5 --out results/escape/n1e5_s$S
python scripts/escape_analyze.py results/escape/n1e5_s$S
done
python scripts/escape_roll.py --ckpt $CK --n 100000 --seed 4738 --steps 1000 --arms T16 --resume --ckpt-every 250 --out results/escape/n1e5_s4738
echo DONE
