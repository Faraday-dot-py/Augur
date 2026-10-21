#!/bin/bash
#SBATCH --job-name=augur-esc-C
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-esc-C-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_escape"
cd "$HOME/augur_escape"
tar xzf "$HOME/escape_code3.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
mkdir -p results/escape
python scripts/escape_tracer_force.py --ckpt $CK --n 20000 --seed 4738 --arms S1,T4,T8,T8w,T16,T8nf,T8old --out results/escape/tracer_force_s4738.json
echo DONE
