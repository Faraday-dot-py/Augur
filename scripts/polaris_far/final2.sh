#!/bin/bash
#SBATCH --job-name=augur-far-final2
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-final2-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
REG=nr100,bh300
M=24
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel old --regimes $REG --per-mult $M --out-dir results/cons96_old
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel old --regimes $REG --per-mult $M --out-dir results/cons96_old2
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel file:kern/r1b_mlp.pt --regimes $REG --per-mult $M --out-dir results/cons96_r1b
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel file:kern/r2b_tail.pt --regimes $REG --per-mult $M --out-dir results/cons96_tail
echo "== control vs control rerun (noise floor)"
python scripts/summarize_cons_compare.py results/cons96_old results/cons96_old2 nr100 bh300
echo "== control vs r1b"
python scripts/summarize_cons_compare.py results/cons96_old results/cons96_r1b nr100 bh300
echo "== control vs tail"
python scripts/summarize_cons_compare.py results/cons96_old results/cons96_tail nr100 bh300
echo DONE
