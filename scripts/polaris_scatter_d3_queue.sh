#!/bin/bash
#SBATCH --job-name=augur-d3-cellgraph
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=augur-d3-cellgraph-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-d3"
export PYTHONPATH=.
CKPT=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
python scripts/scatter_opt_test.py --ckpt "$CKPT" --opts "" --train-opts "default" --ns 2 --cell-ns 1000,3000,10000,30000 2>&1 | grep -v Warning
python scripts/scatter_opt_bench.py --mode infer --ckpt "$CKPT" --ns 1000,10000,30000,100000 --opt default --tag d3_auto 2>&1 | grep -v Warning
python scripts/scatter_opt_bench.py --mode infer --ckpt "$CKPT" --ns 100000 --opt default,-auto,cell --tag d3_cellonly 2>&1 | grep -v Warning
