#!/bin/bash
#SBATCH --job-name=augur-far-aug3
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-aug3-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
R1B=file:kern/r1b_mlp.pt
E=scripts/scatter_far_eval.py
python $E --ckpt $CKPT --phase farerr --kernels $R1B --far-c 4,8,16 --seeds 9000 --nt 8 --n 19200 --sigma 64 --out results/far_aug3_farerr64.json
python scripts/scatter_far_finetune_joint.py --ckpt $CKPT --init kern/r1_mlp.pt --out kern/j1.pt --iters 600 --anchor 0
python scripts/scatter_far_finetune_joint.py --ckpt $CKPT --init kern/r1_mlp.pt --out kern/j2.pt --iters 600 --anchor 100
python $E --ckpt $CKPT --phase table --kernels file:kern/j1.pt,file:kern/j2.pt --out results/far_aug3_table.json
python $E --ckpt $CKPT --phase wide --kernels file:kern/j1.pt,file:kern/j2.pt --far-c 4 --variants far --out results/far_aug3_joint16.json
python $E --ckpt $CKPT --phase wide --kind two --sigma 16 --kernels file:kern/j1.pt,file:kern/j2.pt --far-c 4 --variants far --out results/far_aug3_joint_two.json
echo DONE
