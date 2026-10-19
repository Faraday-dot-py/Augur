#!/bin/bash
#SBATCH --job-name=el-smoke
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=el-smoke-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
ROOT=$HOME/augur-el
mkdir -p $ROOT
cd $ROOT
tar xf $HOME/el_code.tar
mkdir -p results checkpoints
export PYTHONPATH=.
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
run() {
  tag=$1; shift
  rm -rf checkpoints/sm_$tag results/sm_$tag
  python scripts/train_scatter_field.py --exp E --variant ms_kp_pot_v_g128 --mode train --init-ckpt $E --train 200 --steps 50 --iters 200 --batch 16 --k-start 20 --k-end 20 --lr 1e-4 --device cuda --seed 4738 --rloss euclid --log-every 10 --out-dir results/sm_$tag --ckpt-dir checkpoints/sm_$tag "$@" 2>&1 | grep -v Warning
}
echo "=== replay"; run replay --replay-nr100 64 --replay-bh300 64 --replay-frac 0.8 --iters 60
echo SMOKE DONE
