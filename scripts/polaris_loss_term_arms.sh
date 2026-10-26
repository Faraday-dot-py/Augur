#!/bin/bash
#SBATCH --job-name=augur-loss-term-arms
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-loss-term-arms-%j.log
# Three short fine-tunes (control, local work-bias, clump-centre) of the matched-60k tier 0 as concurrent processes on one GPU, then the scoring evals.
# Resumes from <tag>.pt checkpoints (saved every 250 iterations and on SIGTERM); finished outputs are skipped on requeue. Usage: SEED=4738 sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED="${SEED:-4738}"
W=$HOME/augur_loss_terms
mkdir -p $W && cd $W && tar xzf $HOME/loss_terms_code.tgz
export PYTHONPATH=$W
O=results/loss_term_arms/seed$SEED
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur_tiered_c2_tier0_matched_60k.pt
FB="ext=6,extapply,taper=8,bnd,halo=8"
declare -A EXTRA=( [control]="" [local-work-bias]="--work-bias" [clump-centre]="--coherent" )
PIDS=()
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null; wait; exit 143' TERM
T0=$(date +%s)
for arm in control local-work-bias clump-centre; do
  tag=tier0-finetune-$arm-seed$SEED
  [ -f $O/${tag}_eval.json ] && continue
  python scripts/train_tier0_harvest_loss_terms.py ${EXTRA[$arm]} --data $HOME/t0n/data4 --pattern "hv3_s*.npz" --replay 0.5 --lr 1e-4 --iters 4000 \
    --k-start 12 --k-end 20 --t0 "$FB" --init-ckpt $INIT --seed $SEED --tag $tag --out-dir $O --baseline $E > $O/$tag.train.log 2>&1 &
  PIDS+=($!)
done
wait
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
PIDS=()
for arm in control local-work-bias clump-centre; do
  tag=tier0-finetune-$arm-seed$SEED
  CK=$O/$tag.pt
  [ -f $O/${tag}_eval.json ] || { echo "$tag training incomplete"; continue; }
  (
    [ -f $O/${tag}_diag.json ] || python scripts/exp12_diag.py --s1 $CK --tier0 $E --arms OR --out $O/${tag}_diag.json > $O/$tag.diag.log 2>&1
    [ -f $O/${tag}_hier.json ] || python scripts/exp12_hier.py --mode eval --arms OR --s1 $CK --tier0 $E --ckpt-dir $O/hierck --chunk 64 --out $O/${tag}_hier.json > $O/$tag.hier.log 2>&1
    [ -f $O/${tag}_tiling.json ] || python scripts/tier0_tiling_cmp.py --models $arm=$CK --e $E --out $O/${tag}_tiling.json > $O/$tag.tiling.log 2>&1
    [ -f $O/${tag}_hext.json ] || python scripts/hetero_ext_checks.py --ckpt $CK --tag $tag --out $O/${tag}_hext.json > $O/$tag.hext.log 2>&1
  ) &
  PIDS+=($!)
done
wait
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
python scripts/summarize_loss_term_arms.py $O tier0-finetune-control-seed$SEED tier0-finetune-local-work-bias-seed$SEED tier0-finetune-clump-centre-seed$SEED --out $O/summary.json
echo ALLDONE
