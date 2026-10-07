#!/bin/bash
#SBATCH --job-name=augur-scatter-d3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=augur-scatter-d3-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-d3"
export PYTHONPATH=.
CKPT=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/scatter_opt_test.py --ckpt "$CKPT" --opts "default;default,-graph;cell,kcache,fastio,cl,graph" --train-opts "default" --ns 2,100,300,3000 --cell-ns 1000,3000,10000,30000 || echo "OPT_TEST FAILED"
python scripts/scatter_regress.py run --ckpt "$CKPT" --out results/d3_default.json --opt default || echo "REGRESS FAILED"
python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate results/d3_default.json || true
for o in default kcache,fastio,cl,cell kcache,fastio,cl,adapt; do
  python scripts/scatter_opt_bench.py --mode infer --ckpt "$CKPT" --ns 1000,3000,10000,30000,100000 --opt "$o" --tag "d3_infer_${o//,/_}" || echo "BENCH $o FAILED"
done
for o in none default; do
  python scripts/scatter_opt_bench.py --mode train --ckpt "$CKPT" --opt "$o" --tag "d3_train_$o" || echo "TRAIN $o FAILED"
done
