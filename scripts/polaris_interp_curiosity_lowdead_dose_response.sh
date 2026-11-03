#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-lowdead-sae-dose-response
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-lowdead-sae-dose-response-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt scripts/sae_interp.py scripts/sae_ablation.py scripts/sae_ablation_dose_response.py results/sae_interp_topk_dec0_s4738_sae.pt; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec2 --parts mesh --channel-norm --lr 3e-4 --clip 1.0 --mesh-sizes 16,32,64,128 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_dec2_rescaled_s4738.json; echo STEP sae_dec2_rescaled $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec0 --mesh-sizes 16,32,64 --pair-sizes 32,64,128 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec0_s4738.json; echo STEP sae_lowdead_dec0 $?
for k in mesh_32 mesh_64; do
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4738.pt --sae $R/sae_interp_lowdead_dec0_s4738_sae.pt --key $k --out $R/sae_ablation_dose_response_s4738_$k.json; echo STEP dose_response_$k $?
done
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4738.pt --sae $R/sae_interp_topk_dec0_s4738_sae.pt --key mesh_512 --out $R/sae_ablation_dose_response_s4738_mesh_512_original_sae.json; echo STEP dose_response_mesh_512 $?
ls $R/sae_interp_dec2_rescaled* $R/sae_interp_lowdead* $R/sae_ablation_dose_response* | wc -l
