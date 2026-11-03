#!/bin/bash
#SBATCH --job-name=augur-hteval-cpusmoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=augur-hteval-cpusmoke-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur-hteval-cpu"
mkdir -p "$W"
cd "$W"
rm -rf scripts model tests results
cp -r "$HOME/augur/scripts" "$HOME/augur/model" "$HOME/augur/tests" "$HOME/augur/conftest.py" .
cp -r "$HOME/augur-hteval-src/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/tests/." tests/
export PYTHONPATH=.
mkdir -p results/tiered
python scripts/tiered_gen.py truthcheck --device cpu --check-n 4000 --out-json results/tiered/truthcheck_cpu.json
python scripts/tiered_gen.py tier0 --device cpu --kinds all --shards 1 --scenes 8 --batch 8 --steps 4 --out results/tiered/smoke_data
python scripts/tiered_gen.py harvest --device cpu --shards 1 --windows 4 --steps 4 --n-big 1024 --big-steps 8 --out results/tiered/smoke_data
python scripts/tiered_gen.py tier1 --device cpu --shards 1 --scenes 1 --steps 2 --n-range 512 600 --span 128 --lower range8 --out results/tiered/smoke_data
python scripts/tiered_eval.py --device cpu --model oracle --quick --seeds 4738 --out results/tiered/smoke_eval_oracle.json --grid-dir results/tiered/smoke_grids
python scripts/tiered_eval.py --device cpu --model constvel --quick --seeds 4738 --out results/tiered/smoke_eval_constvel.json --grid-dir results/tiered/smoke_grids
CK="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
python scripts/tiered_eval.py --device cpu --model "scatter:$CK" --quick --seeds 4738 --out results/tiered/smoke_eval_scatter.json --grid-dir results/tiered/smoke_grids
ls -la results/tiered results/tiered/smoke_data results/tiered/smoke_grids
