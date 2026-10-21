#!/bin/bash
#SBATCH --job-name=augur-esc-D
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=augur-esc-D-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_escape"
cd "$HOME/augur_escape"
tar xzf "$HOME/escape_code4.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
P=50,100,200,300,400,500,600
for S in 4738 9000 12000; do
python scripts/escape_roll.py --ckpt $CK --n 20000 --nt 2000 --seed $S --steps 100 --arms S1in,T4in,T8in --truth --resume --snap-every 10 --out results/escape/tracer_s$S
python scripts/escape_analyze.py results/escape/tracer_s$S
done
for S in 4738 9000 12000; do
python scripts/escape_roll.py --ckpt $CK --n 20000 --seed $S --steps 1000 --arms S1in,T8in,T4in --truth --resume --probe-ticks $P --out results/escape/n2e4_s$S
python scripts/escape_analyze.py results/escape/n2e4_s$S
done
echo DONE
