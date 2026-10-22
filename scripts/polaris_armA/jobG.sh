#!/bin/bash
#SBATCH --job-name=augur-armA-G
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=augur-armA-G-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_armA"
cd "$HOME/augur_armA"
tar xzf "$HOME/armA_code.tgz"
export PYTHONPATH=.
CK=$HOME/augur_far/kern/E_frozen.pt
mkdir -p results/armA_gate
[ -e results/armA_gate/reg_E.json ] || python scripts/scatter_regress.py run --ckpt $CK --out results/armA_gate/reg_E.json
for V in d4 c4; do
[ -e results/armA_gate/reg_$V.json ] || python scripts/scatter_regress.py run --ckpt $CK --opt default,sym$V --out results/armA_gate/reg_$V.json
python scripts/scatter_regress.py compare --baseline results/armA_gate/reg_E.json --candidate results/armA_gate/reg_$V.json || true
for R in nr100 bh300; do
[ -e results/armA_gate/cons_$V/$R.npz ] || python scripts/scatter_conservation.py run --ckpt $CK --opt default,sym$V --kernel old --regimes $R --per-mult 16 --out-dir results/armA_gate/cons_$V
done
done
CON=$HOME/augur_far/results
for V in d4 c4; do
echo "== stock E (cons64_old) vs sym$V"
python scripts/summarize_cons_compare.py $CON/cons64_old results/armA_gate/cons_$V nr100 bh300
done
echo DONE
