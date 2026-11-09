#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-large-pool-outliers-tie-aware-yukawa
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --nice=100
#SBATCH --output=augur-interp-curiosity-large-pool-outliers-tie-aware-yukawa-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py scripts/sae_ablation_dose_response.py scripts/sae_outlier_cells.py scripts/yukawa_lambda_protocol.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
python scripts/yukawa_lambda_protocol.py --n 1024 --seeds 4741,4742,4743 --out $R/yukawa_lambda_protocol_tie_aware_fresh_seeds.json; echo STEP yukawa_tie_aware $?
python scripts/sae_outlier_cells.py --ckpt ckpt/main_s4738.pt --layer dec3 --sae $R/sae_interp_lowdead_dec3_channelnorm_s4738_sae.pt --key mesh_64 --out $R/sae_outlier_cells_dec3_channelnorm_s4738_mesh_64.json; echo STEP outlier_mesh_64 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec0 --parts mesh --mesh-sizes 128,256 --scenes 24 --heldout-scenes 6 --out $R/sae_interp_lowdead_dec0_s4738_large_dictionaries.json; echo STEP large_dictionaries $?
for k in mesh_128 mesh_256; do
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4738.pt --sae $R/sae_interp_lowdead_dec0_s4738_large_dictionaries_sae.pt --key $k --pool-size 50 --act-scenes 24 --out $R/sae_ablation_dose_response_large_pool_s4738_$k.json; echo STEP dose_response_large_pool_$k $?
done
python scripts/sae_ablation_dose_response.py --ckpt ckpt/main_s4738.pt --sae $R/sae_interp_topk_dec0_s4738_sae.pt --key mesh_512 --pool-size 50 --act-scenes 24 --out $R/sae_ablation_dose_response_large_pool_s4738_mesh_512_original_sae.json; echo STEP dose_response_large_pool_mesh_512 $?
ls -l $R/*tie_aware* $R/sae_outlier* $R/*large*
