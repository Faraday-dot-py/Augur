#!/bin/bash
#SBATCH --job-name=augur-esc-G
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=augur-esc-G-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_escape"
cd "$HOME/augur_escape"
tar xzf "$HOME/escape_code6.tgz"
export PYTHONPATH=.
CK=$HOME/edge_ck/ghost_lsq.pt
P=100,200,300
run() {
  name=$1; shift
  python scripts/escape_roll.py --ckpt $CK --n 20000 --seed 4738 --steps 500 --arms S1,T8,S1nm --resume --probe-ticks $P --out results/escape/geom_$name "$@"
  python scripts/escape_analyze.py results/escape/geom_$name
}
run rot90 --rot 1
run rot180 --rot 2
run halfcell --shift 0.25,0.25
run quartercell --shift 0.125,0.0
run int6 --shift 3.0,3.0
run off --shift 3.25,-2.75
echo DONE
