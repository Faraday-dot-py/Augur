#!/bin/bash
#SBATCH --job-name=augur-hteval-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --output=augur-hteval-smoke-%j.log

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
nvidia-smi --query-gpu=name,memory.used --format=csv
python -m pytest tests/test_tiered_harness.py -q 2>&1 | tail -40
python scripts/tiered_gen.py truthcheck --out-json results/tiered/truthcheck.json
python scripts/tiered_gen.py tier0 --kinds all --shards 1 --scenes 16 --steps 8 --out results/tiered/smoke_data
python scripts/tiered_gen.py harvest --shards 1 --windows 8 --steps 8 --n-big 2048 --big-steps 24 --out results/tiered/smoke_data
python scripts/tiered_gen.py tier1 --shards 1 --scenes 2 --steps 4 --n-range 512 1024 --span 128 --lower range8 --out results/tiered/smoke_data
python scripts/tiered_eval.py --model oracle --quick --seeds 4738 --out results/tiered/smoke_eval_oracle.json --grid-dir results/tiered/smoke_grids
