#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-layer-stats-neighbour-layers
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-layer-stats-neighbour-layers-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py scripts/layer_activation_stats.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
python scripts/layer_activation_stats.py --ckpt ckpt/main_s4738.pt --out $R/layer_activation_stats_s4738.json; echo STEP layer_stats_4738 $?
python scripts/layer_activation_stats.py --ckpt ckpt/main_s4739.pt --out $R/layer_activation_stats_s4739checkpoint.json; echo STEP layer_stats_4739 $?
for L in dec1 dec3; do
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer $L --parts mesh --mesh-sizes 16,32,64 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_${L}_s4738.json; echo STEP sae_lowdead_$L $?
done
