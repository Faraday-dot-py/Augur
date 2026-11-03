#!/bin/bash
#SBATCH --job-name=augur-far-aug2
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-aug2-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
R1B=file:kern/r1b_mlp.pt
TAIL=file:kern/r2b_tail.pt
E=scripts/scatter_far_eval.py
python scripts/scatter_far_train_kernel.py --ckpt $CKPT --arch tail --lo 32 --steps 8000 --lr 2e-3 --out kern/r2b_tail.pt
python $E --ckpt $CKPT --phase table --kernels file:kern/r1_mlp.pt,$R1B,$TAIL --out results/far_aug2_table.json
python $E --ckpt $CKPT --phase wide --kernels $TAIL --far-c 4 --variants far,exactfar --out results/far_aug2_tail16.json
python $E --ckpt $CKPT --phase wide --kind two --sigma 16 --kernels $R1B --far-c 4 --variants exactfar --out results/far_aug2_two_exactfar.json
python $E --ckpt $CKPT --phase wide --kind uniform --kernels $R1B --far-c 4 --variants exactfar --out results/far_aug2_uniform_exactfar.json
python $E --ckpt $CKPT --phase farerr --kernels $R1B --far-c 4,8,16 --seeds 9000 --nt 8 --n 19200 --sigma 64 --out results/far_aug2_farerr64.json
python $E --ckpt $CKPT --phase farerr --kernels $R1B,$TAIL --far-c 4 --kind two --sigma 16 --out results/far_aug2_farerr_two.json
python $E --ckpt $CKPT --phase wide --nt 8 --n 19200 --sigma 64 --kernels $TAIL --far-c 4 --variants far --out results/far_aug2_tail64.json
python $E --ckpt $CKPT --phase wide --nt 8 --n 19200 --sigma 64 --kernels $R1B --far-c 4 --variants exactfar --seeds 9000 --out results/far_aug2_exactfar64.json
echo DONE
