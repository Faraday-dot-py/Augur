#!/bin/bash
#SBATCH --job-name=augur-far-aug1
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=augur-far-aug1-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
R1=file:kern/r1_mlp.pt
R1B=file:kern/r1b_mlp.pt
E=scripts/scatter_far_eval.py
for tag in _oldk _r1k; do python scripts/summarize_scatter_tiling.py $tag > results/scatter_tiling_summary$tag.txt; done
python scripts/scatter_far_train_kernel.py --ckpt $CKPT --arch mlp --out kern/r1b_mlp.pt --steps 30000 --rmax 1500
python $E --ckpt $CKPT --phase table --kernels $R1,$R1B --out results/far_aug1_table.json
python $E --ckpt $CKPT --phase wide --kernels $R1B --far-c 4 --variants far --out results/far_aug1_wide16_r1b.json
python $E --ckpt $CKPT --phase wide --nt 8 --n 19200 --sigma 64 --kernels old,$R1,$R1B --far-c 4,8,16 --variants far,halo --out results/far_aug1_wide64.json
python $E --ckpt $CKPT --phase wide --kind two --sigma 16 --kernels old,$R1B --far-c 4 --variants far,halo --out results/far_aug1_two.json
python $E --ckpt $CKPT --phase wide --kind uniform --kernels old,$R1B --far-c 4 --variants far,halo --out results/far_aug1_uniform.json
python $E --ckpt $CKPT --phase wide --steps 100 --kernels old,$R1B --far-c 4 --variants far,halo --out results/far_aug1_long100.json
echo DONE
