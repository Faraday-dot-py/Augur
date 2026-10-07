#!/bin/bash
#SBATCH --job-name=augur-d3-followup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=augur-d3-followup-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-d3"
export PYTHONPATH=.
CKPT=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
python scripts/scatter_opt_test.py --ckpt "$CKPT" --opts "" --train-opts "" --ns 2 --cell-ns 1000,3000,10000,30000 2>&1 | grep -E "PPCAP|CELLGRAPH"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor-cache-d3 TRITON_CACHE_DIR=$HOME/.triton-cache-d3
H=$HOME/py312hdr_conda
SRC=/data/data03/home/davidmalone/conda_caches/conda_pkgs/python-3.12.14-h5f976f7_3_cpython/include/python3.12
[ -r "$H/Python.h" ] || { mkdir -p "$H" && cp -r "$SRC"/. "$H"/; }
export C_INCLUDE_PATH=$H CPLUS_INCLUDE_PATH=$H
for mode in default max-autotune-no-cudagraphs; do
  rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  echo "=== compile mode $mode (own-home header copy)"
  timeout 900 python scripts/probe_compile.py --ckpt "$CKPT" --mode $mode --ns 1000,3000,10000 2>&1 | grep -v Warning | tail -30
done
