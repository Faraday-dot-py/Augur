#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-dec3-channel-norm-both-checkpoints
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-dec3-channel-norm-both-checkpoints-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec3 --parts mesh --channel-norm --lr 3e-4 --clip 1.0 --mesh-sizes 16,32,64 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec3_channelnorm_s4738.json; echo STEP dec3_channelnorm_4738 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4739.pt --layer dec3 --parts mesh --channel-norm --lr 3e-4 --clip 1.0 --mesh-sizes 16,32,64 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec3_channelnorm_s4739checkpoint.json; echo STEP dec3_channelnorm_4739 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4739.pt --layer dec2 --parts mesh --mesh-sizes 16,32,64 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec2_s4739checkpoint.json; echo STEP dec2_4739 $?
ls -l $R/sae_interp_lowdead_dec3_channelnorm* $R/sae_interp_lowdead_dec2_s4739checkpoint*
