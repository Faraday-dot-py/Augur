#!/bin/bash
#SBATCH --job-name=augur-hteval-base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --output=augur-hteval-base-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur-hteval"
mkdir -p "$W/results/tiered"
cd "$W"
rm -rf scripts model tests
cp -r "$HOME/augur/scripts" "$HOME/augur/model" "$HOME/augur/tests" "$HOME/augur/conftest.py" .
cp -r "$HOME/augur-hteval-src/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/tests/." tests/
export PYTHONPATH=.
CK="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
nvidia-smi --query-gpu=name,memory.used --format=csv
python scripts/tiered_eval.py --model "scatter:$CK" --out results/tiered/eval_scatterE.json --grid-dir results/tiered/grids || echo "FAILED scatterE"
python scripts/tiered_eval.py --model constvel --out results/tiered/eval_constvel.json --grid-dir results/tiered/grids || echo "FAILED constvel"
python scripts/tiered_eval.py --model oracle --out results/tiered/eval_oracle.json --grid-dir results/tiered/grids || echo "FAILED oracle"
