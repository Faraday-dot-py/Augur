#!/bin/bash
#SBATCH --job-name=augur-armA-2
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=augur-armA-2-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_armA"
cd "$HOME/augur_armA"
tar xzf "$HOME/armA_code.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
SRC=$HOME/augur_escape/results/escape
P=50,100,200,300,400,500,600
for S in 9000 12000; do
D=results/armA/n2e4_s$S
mkdir -p $D
for f in truth.npz truth.json core_ids.npy S1.npz S1.json; do [ -e $D/$f ] || cp $SRC/n2e4_s$S/$f $D/; done
python scripts/escape_roll.py --ckpt $CK --n 20000 --seed $S --steps 1000 --arms S1d4,S1c4,T8d4 --truth --resume --probe-ticks $P --out $D
python scripts/escape_analyze.py $D
done
echo DONE
