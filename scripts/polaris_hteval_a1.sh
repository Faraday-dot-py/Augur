#!/bin/bash
#SBATCH --job-name=augur-hteval-a1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --output=augur-hteval-a1-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur-hteval"
mkdir -p "$W/results/tiered"
cd "$W"
rm -rf scripts model tests
cp -r "$HOME/augur/scripts" "$HOME/augur/model" "$HOME/augur/tests" "$HOME/augur/conftest.py" .
cp -r "$HOME/augur-hteval-src/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/tests/." tests/
cp -r "$HOME/augur-hteval-src/extra/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/extra/model/." model/
export PYTHONPATH=.
R=results/tiered
R=results/tiered
A1="$HOME/augur-tier0/results/tier0_ext/armA1.pt"
python scripts/tiered_eval.py --model "t0ext:$A1" --groups bias,seams,gravity,boundaries,exposure,generalisation --out $R/eval_t0ext_A1.json --grid-dir $R/grids_t0ext_A1 || echo "FAILED eval A1"
python scripts/tiered_diag_blowup.py --model "t0ext:$A1" --out $R/diag_blowup_A1.json || echo "FAILED diag A1"
echo A1DONE
