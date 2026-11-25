#!/bin/bash
#SBATCH --job-name=augur-far-r1
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-r1-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
python scripts/scatter_far_train_kernel.py --ckpt $CKPT --arch mlp --out kern/r1_mlp.pt --steps 30000
python scripts/scatter_far_eval.py --ckpt $CKPT --phase table --kernels file:kern/r1_mlp.pt --out results/far_r1_table.json
python scripts/scatter_far_eval.py --ckpt $CKPT --phase wide --kernels file:kern/r1_mlp.pt --far-c 4 --variants far,exactfar --out results/far_r1_wide.json
python scripts/scatter_far_eval.py --ckpt $CKPT --phase farerr --kernels file:kern/r1_mlp.pt --far-c 4 --out results/far_r1_farerr.json
