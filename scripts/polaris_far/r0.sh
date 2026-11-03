#!/bin/bash
#SBATCH --job-name=augur-far-r0
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=augur-far-r0-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
nvidia-smi --query-gpu=name,memory.used --format=csv
[ -f kern/E_frozen.pt ] || cp checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt kern/E_frozen.pt
sha256sum kern/E_frozen.pt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
CKPT=kern/E_frozen.pt
python scripts/scatter_far_eval.py --ckpt $CKPT --phase table --kernels old,oldf,clamp40_zero,clamp40_inv2,clamp64_zero,clamp64_inv2,oracle64 --out results/far_r0_table.json
python scripts/scatter_far_eval.py --ckpt $CKPT --phase wide --kernels old,oldf,clamp40_zero,clamp40_inv2,clamp64_zero,clamp64_inv2,oracle64 --far-c 4 --variants far,halo --out results/far_r0_wide.json
python scripts/scatter_far_eval.py --ckpt $CKPT --phase wide --kernels oracle64 --far-c 2,8 --variants far --out results/far_r0_wide_farc.json
python scripts/scatter_far_eval.py --ckpt $CKPT --phase wide --kernels oracle64,old --far-c 4 --variants exactfar --out results/far_r0_wide_exactfar.json
python scripts/scatter_far_eval.py --ckpt $CKPT --phase farerr --kernels old,oracle64,clamp64_inv2,clamp40_inv2 --far-c 2,4,8 --out results/far_r0_farerr.json
