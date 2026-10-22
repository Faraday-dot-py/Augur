#!/bin/bash
#SBATCH --job-name=augur-far-final1
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-final1-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
R1B=file:kern/r1b_mlp.pt
TAIL=file:kern/r2b_tail.pt
E=scripts/scatter_far_eval.py
python $E --ckpt $CKPT --phase time --kernels oldf,oldf+cache,$R1B,$R1B+cache,$TAIL,$TAIL+cache --out results/far_final1_time.json
run_tiling() {
  kern=$1; tag=$2
  python scripts/scatter_tiling_eval.py --ckpt $CKPT --kernel $kern --phase force --out results/scatter_tiling_force$tag.json
  python scripts/scatter_tiling_eval.py --ckpt $CKPT --kernel $kern --phase roll --out results/scatter_tiling_roll$tag.json
  python scripts/summarize_scatter_tiling.py $tag > results/scatter_tiling_summary$tag.txt
}
run_tiling $R1B _r1bk
run_tiling $TAIL _tailk
echo DONE
