#!/bin/bash
#SBATCH --job-name=augur-mixed-law-pow1-amp-cap-scoring-eval
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterany:7296
#SBATCH --output=augur-mixed-law-pow1-amp-cap-scoring-eval-%j.log
# Scoring eval for job 7296 (experiment-log.md:3787 pre-registered line). Inference only, 1 GPU, no training.
# Runs in the 7296 work dir so scatter_multi.py is the copy the checkpoint was trained with. Scored locally by scripts/score_mixed_law_pow1_amp_cap.py.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur-mixed-law-pow1-amp-cap-baseline-scatter-field
U=$HOME/augur-mixed-law-upload
NEW=$W/ckpt/mixed_pow1ampcap_s4738.pt
CMP=$HOME/augur_nfr/ckpt/main_s4739.pt
for f in $NEW $W/ckpt/mixed_pow1ampcap_s4738.done $CMP $W/kern/E_frozen.pt $W/scripts/sae_interp.py $W/scripts/scatter_multi.py $U/scatter_multi_eval.py $U/layer_activation_stats.py $U/render_multi_rollout_grid.py; do
  test -s $f || { echo MISSING $f; exit 2; }
done
for f in scatter_multi_eval.py layer_activation_stats.py render_multi_rollout_grid.py; do cp $U/$f $W/scripts/$f || exit 2; done
cd $W || exit 2
export PYTHONPATH=.
md5sum scripts/scatter_multi.py scripts/scatter_multi_eval.py scripts/layer_activation_stats.py scripts/render_multi_rollout_grid.py $NEW $CMP
nvidia-smi --query-gpu=name,memory.total --format=csv
R=results/pow1_amp_cap_scoring_eval
mkdir -p $R
python scripts/scatter_multi_eval.py --ckpt pow1cap_s4738=$NEW --ckpt main_s4739=$CMP --e-ckpt kern/E_frozen.pt --cases all --out $R/eval.json
rc1=$?; echo STEP scatter_multi_eval $rc1
python scripts/layer_activation_stats.py --ckpt $NEW --out $R/layer_stats_pow1cap_s4738.json
rc2=$?; echo STEP layer_stats $rc2
python scripts/render_multi_rollout_grid.py dump --ckpt pow1cap_s4738=$NEW --ckpt main_s4739=$CMP --e-ckpt kern/E_frozen.pt --n 48 --out $R/multi_traj.npz
rc3=$?; echo STEP rollout_dump $rc3
for f in $R/eval.json $R/layer_stats_pow1cap_s4738.json $R/multi_traj.npz; do test -s $f || { echo MISSING OUTPUT $f; exit 1; }; done
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] || { echo FAILED $rc1 $rc2 $rc3; exit 1; }
echo ALLDONE
