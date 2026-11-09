#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-lowdead-second-checkpoint-sae-seed
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-lowdead-second-checkpoint-sae-seed-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py scripts/sae_ablation_dose_response.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
python scripts/sae_interp.py --ckpt ckpt/main_s4739.pt --layer dec0 --mesh-sizes 16,32,64 --pair-sizes 32,64,128 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec0_s4739checkpoint.json; echo STEP lowdead_dec0_second_checkpoint $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec0 --sae-seed 4739 --mesh-sizes 16,32,64 --pair-sizes 32,64,128 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec0_s4738_saeseed4739.json; echo STEP lowdead_dec0_sae_seed $?
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4739.pt --sae $R/sae_interp_lowdead_dec0_s4739checkpoint_sae.pt --key mesh_64 --out $R/sae_ablation_dose_response_s4739checkpoint_mesh_64.json; echo STEP dose_response_second_checkpoint $?
ls -l $R/sae_interp_lowdead_dec0_s4739checkpoint* $R/sae_interp_lowdead_dec0_s4738_saeseed4739* $R/sae_ablation_dose_response_s4739checkpoint*
