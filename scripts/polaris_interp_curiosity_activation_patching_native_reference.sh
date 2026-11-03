#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-activation-patching-native-reference
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-activation-patching-native-reference-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt scripts/sae_interp.py scripts/sae_activation_patching.py results/sae_interp_lowdead_dec1_s4738_sae.pt; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
D=$R/sae_interp_lowdead_dec0_s4738_heldout16_512_dictionary
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec0 --parts mesh --mesh-sizes 512 --scenes 48 --heldout-scenes 16 --out $D.json; echo STEP sae_dec0_512 $?
python scripts/sae_activation_patching.py --ckpt ckpt/main_s4738.pt --layer dec0 --sae ${D}_sae.pt --key mesh_512 --act-scenes 48 --heldout-rank 16 --scenes 24 --ks 1,5,10,20,50 --extended --out $R/sae_activation_patching_native_reference_dec0_s4738_mesh_512.json; echo STEP patch_dec0 $?
python scripts/sae_activation_patching.py --ckpt ckpt/main_s4738.pt --layer dec1 --sae $R/sae_interp_lowdead_dec1_s4738_sae.pt --key mesh_64 --act-scenes 24 --heldout-rank 6 --scenes 24 --ks 1,5,10,20,50 --extended --out $R/sae_activation_patching_native_reference_dec1_s4738_mesh_64.json; echo STEP patch_dec1 $?
ls -l $R/sae_activation_patching_native_reference* $D*
