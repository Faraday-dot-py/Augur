#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-dose-response-replication-activation-patching
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-dose-response-replication-activation-patching-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py scripts/sae_ablation_dose_response.py scripts/sae_activation_patching.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
D2=$R/sae_interp_lowdead_dec2_s4739checkpoint_large_dictionaries
D0=$R/sae_interp_lowdead_dec0_s4738_heldout16_large_dictionaries
python scripts/sae_interp.py --ckpt ckpt/main_s4739.pt --layer dec2 --parts mesh --mesh-sizes 128,256 --scenes 24 --heldout-scenes 6 --out $D2.json; echo STEP sae_dec2_4739 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec0 --parts mesh --mesh-sizes 128,256 --scenes 48 --heldout-scenes 16 --out $D0.json; echo STEP sae_dec0_4738_heldout16 $?
for k in mesh_128 mesh_256; do
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4739.pt --layer dec2 --sae ${D2}_sae.pt --key $k --draws 10 --pool-size 50 --act-scenes 24 --heldout-rank 6 --scenes 32 --out $R/sae_ablation_dose_response_dec2_s4739checkpoint_$k.json; echo STEP dose_dec2_4739_$k $?
done
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4738.pt --layer dec0 --sae ${D0}_sae.pt --key mesh_256 --draws 10 --pool-size 50 --act-scenes 48 --heldout-rank 16 --scenes 32 --out $R/sae_ablation_dose_response_dec0_s4738_heldout16_mesh_256.json; echo STEP dose_dec0_4738 $?
python scripts/sae_activation_patching.py --ckpt ckpt/main_s4739.pt --layer dec2 --sae ${D2}_sae.pt --key mesh_256 --act-scenes 24 --heldout-rank 6 --scenes 16 --out $R/sae_activation_patching_dec2_s4739checkpoint_mesh_256.json; echo STEP patch_dec2_4739 $?
python scripts/sae_activation_patching.py --ckpt ckpt/main_s4738.pt --layer dec0 --sae ${D0}_sae.pt --key mesh_256 --act-scenes 48 --heldout-rank 16 --scenes 16 --out $R/sae_activation_patching_dec0_s4738_mesh_256.json; echo STEP patch_dec0_4738 $?
ls -l $R/*replication* $R/*heldout16* $R/*large_dictionaries* $R/sae_activation_patching* 2>/dev/null
