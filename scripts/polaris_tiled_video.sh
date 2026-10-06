#!/bin/bash
#SBATCH --job-name=augur-tiled-video
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=augur-tiled-video-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
pip install -q -r requirements.txt

mkdir -p results
python scripts/tiled_video.py --ticks 3000 --record-every 3 --density-only --out results/tiled_video_10m_3000.npz
