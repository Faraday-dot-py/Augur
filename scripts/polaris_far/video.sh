#!/bin/bash
#SBATCH --job-name=augur-far-video
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=augur-far-video-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur_far"
export PYTHONPATH=.
CKPT=kern/E_frozen.pt
python scripts/scatter_far_video.py --ckpt $CKPT --kernel file:kern/r1b_mlp.pt --out results/farvideo_gauss.npz --kind gauss --steps 100
python scripts/scatter_far_video.py --ckpt $CKPT --kernel file:kern/r1b_mlp.pt --out results/farvideo_two.npz --kind two --sigma 16 --steps 100
echo DONE
