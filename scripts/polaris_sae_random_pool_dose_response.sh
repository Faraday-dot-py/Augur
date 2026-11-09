#!/bin/bash
#SBATCH --job-name=augur-sae-random-pool-dose-response
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --nice=100
#SBATCH --output=augur-sae-random-pool-dose-response-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
R=results
fail=0
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt scripts/sae_interp.py scripts/sae_ablation.py scripts/sae_ablation_dose_response.py scripts/sae_ablation_random_pool_dose_response.py; do test -f $f || { echo MISSING $f; exit 2; }; done
run() {
    local tag=$1 ckpt=$2 layer=$3 sae=$4 key=$5 act=$6 held=$7
    local out=$R/sae_random_pool_dose_response_${tag}_$key.json
    test -f $sae || { echo MISSING $sae; fail=1; return; }
    if grep -q '"complete": true' $out 2>/dev/null; then echo SKIP_DONE $out; return; fi
    python scripts/sae_ablation_random_pool_dose_response.py --ckpt $ckpt --layer $layer --sae $sae --key $key --act-scenes $act --heldout-rank $held --draws 200 --pool-size 50 --scenes 8 --seed 4738 --out $out
    local rc=$?
    echo STEP ${tag}_$key $rc
    [ $rc -eq 0 ] || fail=1
}
run dec0_s4738 ckpt/main_s4738.pt dec0 $R/sae_interp_lowdead_dec0_s4738_large_dictionaries_sae.pt mesh_256 24 0
run dec0_s4738_original_sae ckpt/main_s4738.pt dec0 $R/sae_interp_topk_dec0_s4738_sae.pt mesh_512 24 0
run dec0_s4738 ckpt/main_s4738.pt dec0 $R/sae_interp_lowdead_dec0_s4738_large_dictionaries_sae.pt mesh_128 24 0
run dec0_s4738 ckpt/main_s4738.pt dec0 $R/sae_interp_lowdead_dec0_s4738_sae.pt mesh_64 24 0
run dec0_s4739checkpoint ckpt/main_s4739.pt dec0 $R/sae_interp_lowdead_dec0_s4739checkpoint_sae.pt mesh_64 24 0
run dec2_s4739checkpoint ckpt/main_s4739.pt dec2 $R/sae_interp_lowdead_dec2_s4739checkpoint_large_dictionaries_sae.pt mesh_256 24 6
run dec2_s4739checkpoint ckpt/main_s4739.pt dec2 $R/sae_interp_lowdead_dec2_s4739checkpoint_large_dictionaries_sae.pt mesh_128 24 6
run dec0_s4738_heldout16 ckpt/main_s4738.pt dec0 $R/sae_interp_lowdead_dec0_s4738_heldout16_large_dictionaries_sae.pt mesh_256 48 16
ls -l $R/sae_random_pool_dose_response_*
exit $fail
