#!/bin/bash
#SBATCH --job-name=augur-far-cons
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-cons-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
K=file:kern/r1_mlp.pt
REG=nr100,bh300
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel old --regimes $REG --per-mult 16 --out-dir results/cons64_old
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel old --regimes $REG --per-mult 16 --out-dir results/cons64_old2
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel $K --regimes $REG --per-mult 16 --out-dir results/cons64_r1
echo "== control vs control rerun (noise floor)"
python scripts/summarize_cons_compare.py results/cons64_old results/cons64_old2 nr100 bh300
echo "== control vs r1"
python scripts/summarize_cons_compare.py results/cons64_old results/cons64_r1 nr100 bh300
echo DONE
