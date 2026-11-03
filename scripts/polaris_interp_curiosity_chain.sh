#!/bin/bash
#SBATCH --job-name=augur-interp-curiosity-sae-probes-ablation
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --dependency=afterany:4827
#SBATCH --output=augur-interp-curiosity-sae-probes-ablation-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py scripts/activation_probes.py scripts/sae_ablation.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --out $R/sae_interp_topk_dec0_s4738.json; echo STEP sae_main $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --sae-seed 4739 --out $R/sae_interp_topk_dec0_s4738_saeseed4739.json; echo STEP sae_stability $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer dec2 --parts mesh --out $R/sae_interp_topk_dec2_s4738.json; echo STEP sae_dec2 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --layer enc4 --parts mesh --out $R/sae_interp_topk_enc4_s4738.json; echo STEP sae_enc4 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4738.pt --pair-layer l1 --parts pair --out $R/sae_interp_topk_pairl1_s4738.json; echo STEP sae_pair_l1 $?
python scripts/sae_interp.py --ckpt ckpt/main_s4739.pt --out $R/sae_interp_topk_dec0_s4739checkpoint.json; echo STEP sae_second_checkpoint $?
python scripts/activation_probes.py --ckpt ckpt/main_s4738.pt --out $R/activation_probes_s4738.json; echo STEP probes_main $?
python scripts/activation_probes.py --ckpt ckpt/main_s4739.pt --out $R/activation_probes_s4739checkpoint.json; echo STEP probes_second_checkpoint $?
for k in mesh_128 mesh_512; do
python scripts/sae_ablation.py --ckpt ckpt/main_s4738.pt --sae $R/sae_interp_topk_dec0_s4738_sae.pt --key $k --out $R/sae_ablation_s4738_$k.json; echo STEP ablation_$k $?
done
ls $R/sae_* $R/activation_probes_* | wc -l
