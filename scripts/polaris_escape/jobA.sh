#!/bin/bash
#SBATCH --job-name=augur-esc-A
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=augur-esc-A-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_escape"
cd "$HOME/augur_escape"
tar xzf "$HOME/escape_code2.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
P=50,100,200,300,400,500,600
for S in 4738 9000 12000; do
python scripts/escape_roll.py --ckpt $CK --n 20000 --nt 2000 --seed $S --steps 100 --arms S0,S1,T4,T8,T8w,T16 --truth --resume --snap-every 10 --out results/escape/tracer_s$S
python scripts/escape_analyze.py results/escape/tracer_s$S
done
python scripts/escape_roll.py --ckpt $CK --n 20000 --seed 4738 --steps 1000 --arms S1,T8,T4,S1nm,T8nm,S0,S1g,T8w,T4nm,T16 --truth --resume --probe-ticks $P --out results/escape/n2e4_s4738
python scripts/escape_analyze.py results/escape/n2e4_s4738
echo DONE
