#!/bin/bash
#SBATCH --job-name=augur-scatter-recabl-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-scatter-recabl-eval-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
TAG="${TAG:?name}"
VAR="${VAR:?variant}"
CKPT="${CKPT:?ckpt path relative to augur-recabl or absolute}"
cp "$HOME"/recabl_stage/*.py "$HOME/augur-recabl/scripts/"
cd "$HOME/augur-recabl"
export PYTHONPATH=.
OUT=results/recabl_eval/$TAG
mkdir -p $OUT
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
python scripts/scatter_recurrent_ablation_eval.py --ckpt "$CKPT" --variant $VAR --out $OUT/channel_eval.json
python scripts/scatter_regress.py run --ckpt "$CKPT" --variant $VAR --out $OUT/regress.json
# 24 held-out-scene conservation (8 scenes/seed x 9100/9200/9300; 4738 rows ignored in analysis), 1000 ticks
python scripts/scatter_conservation.py run --ckpt "$CKPT" --variant $VAR --opt kcache,fastio,cl,graph --steps 1000 --per-seed 8 --regimes nr10,nr100,bh100,bh300 --out-dir $OUT/cons1k
# long-horizon stability, harness-default scene counts, 1e4 ticks
python scripts/scatter_conservation.py run --ckpt "$CKPT" --variant $VAR --opt kcache,fastio,cl,graph --steps 10000 --regimes nr100,bh300 --out-dir $OUT/cons10k
