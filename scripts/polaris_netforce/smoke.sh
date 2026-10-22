#!/bin/bash
#SBATCH --job-name=augur-nf-smoke
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=augur-nf-smoke-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
rm -rf "$HOME/augur_nf_smoke"; mkdir -p "$HOME/augur_nf_smoke"
cd "$HOME/augur_nf_smoke"
tar xzf "$HOME/nf_code.tgz"
export PYTHONPATH=.
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
nvidia-smi --query-gpu=name,memory.used --format=csv
python scripts/train_scatter_netforce.py --arm Bs --lam 0.1 --init-ckpt $E --train 60 --n-bank 4 --iters 40 --log-every 5 --out-dir results/nf_smoke --ckpt-dir checkpoints/nf_smoke
python scripts/train_scatter_netforce.py --arm Cs --lam 0 --mu 0 --init-ckpt $E --train 60 --n-bank 4 --iters 40 --log-every 5 --out-dir results/nf_smoke --ckpt-dir checkpoints/nf_smoke
python scripts/escape_roll.py --ckpt checkpoints/nf_smoke/Bs.pt --variant ms_kp_pot_v_g128 --n 3000 --seed 4738 --steps 110 --arms S1 --probe-ticks 100 --out results/nf_smoke/roll
python scripts/core_drift_summary.py Bs=results/nf_smoke/roll S1
echo SMOKE DONE
