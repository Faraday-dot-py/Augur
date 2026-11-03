#!/bin/bash
#SBATCH --job-name=augur-far-r1reg
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=augur-far-r1reg-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
K=file:kern/r1_mlp.pt
run_tiling() {
  kern=$1; tag=$2
  python scripts/scatter_tiling_eval.py --ckpt $CKPT --kernel $kern --phase force --out results/scatter_tiling_force$tag.json
  python scripts/scatter_tiling_eval.py --ckpt $CKPT --kernel $kern --phase roll --out results/scatter_tiling_roll$tag.json
  python scripts/scatter_tiling_eval.py --ckpt $CKPT --kernel $kern --phase scale --out results/scatter_tiling_scale$tag.json
  python scripts/summarize_scatter_tiling.py $tag > results/scatter_tiling_summary$tag.txt
}
run_tiling old _oldk
run_tiling $K _r1k
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel old --regimes nr100,bh300 --out-dir results/cons_old
python scripts/scatter_conservation.py run --ckpt $CKPT --kernel $K --out-dir results/cons_r1
for d in cons_old cons_r1; do
  python scripts/scatter_conservation_summarize.py --dir results/$d --out-json results/$d/summary.json --fig-dir results/$d/figs > results/$d/summary.txt
done
echo DONE
