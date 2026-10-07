#!/bin/bash
#SBATCH --job-name=augur-probe-compile
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=augur-probe-compile-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-d3"
export PYTHONPATH=.
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor-cache-d3 TRITON_CACHE_DIR=$HOME/.triton-cache-d3
CKPT=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
run() {
  echo "=== VARIANT $1 (C_INCLUDE_PATH=${C_INCLUDE_PATH:-})"
  rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
  timeout 600 python scripts/probe_compile.py --ckpt "$CKPT" 2>&1 | grep -v Warning | tail -40
}
nvidia-smi --query-gpu=name --format=csv
ls /usr/include/python3.12/Python.h 2>&1
run v0_stock
for d in /opt/carla-0.9.16/pyenv/include/python3.12 /data/data03/home/davidmalone/conda_caches/conda_pkgs/python-3.12.14-h5f976f7_3_cpython/include/python3.12; do
  if [ -r "$d/Python.h" ]; then
    export C_INCLUDE_PATH=$d CPLUS_INCLUDE_PATH=$d
    run "v1_$d"
  else
    echo "unreadable $d"
  fi
done
unset C_INCLUDE_PATH CPLUS_INCLUDE_PATH
H=$HOME/py312hdr
if [ ! -r "$H/Python.h" ]; then
  mkdir -p "$H/src"
  cd "$H/src"
  curl -sL https://www.python.org/ftp/python/3.12.14/Python-3.12.14.tgz | tar xz --strip-components=1 Python-3.12.14/Include
  cp -r Include/* "$H/"
  cp /usr/include/python3.12/pyconfig*.h "$H/"
  cd "$HOME/augur-d3"
fi
ls "$H/Python.h"
export C_INCLUDE_PATH=$H CPLUS_INCLUDE_PATH=$H
run "v2_python_org_headers"
