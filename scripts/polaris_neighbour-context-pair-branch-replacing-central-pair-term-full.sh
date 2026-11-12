#!/bin/bash
# NOT YET AUTHORISED: full run (about 2 GPU-h). Do not submit until stage 1 passes and the supervisor releases it.
#SBATCH --job-name=augur-neighbour-context-pair-branch-full
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --nice=100
#SBATCH --requeue
#SBATCH --dependency=afterany:SLOT_TAIL_PLACEHOLDER
#SBATCH --output=augur-neighbour-context-pair-branch-full-%j.log
# Full run = stage 1 recipe at 4x iterations (6000), same seeds, same scoring on fresh eval seeds 12000/12100/12200. Control iteration count is kept equal to the candidate's.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_pair_context
export PYTHONPATH=.
E=$HOME/E_ms_kp_pot_v_g128.pt
ITERS=6000
SCENES=1500
TRAIN=results/pair_context_full/train
CK=ckpt_full
for f in $E scripts/train_scatter_ft.py scripts/pair_context_eval.py scripts/scatter_field.py; do test -f $f || { echo MISSING $f; exit 2; }; done
mkdir -p $TRAIN $CK
CHILD=0
trap '[ $CHILD -ne 0 ] && kill -TERM $CHILD 2>/dev/null; wait; echo SIGTERM handled; exit 143' TERM
for seed in 4739; do
  for arm in ctrl cand; do
    name=pairctx-$arm-s$seed
    if [ -f $TRAIN/${name}_train.json ]; then echo "cell $name done, skipping"; continue; fi
    if [ $arm = ctrl ]; then fix=ctrl; variant=ms_kp_pot_v_g128; else fix=pairctx; variant=ms_kp_pot_v_g128_pairctx; fi
    resume=""; [ -f $CK/$name.pt ] && resume="--resume"
    python scripts/train_scatter_ft.py --fix $fix --variant $variant --init-ckpt $E --name $name --seed $seed --iters $ITERS --scenes $SCENES \
      --out-dir $TRAIN --ckpt-dir $CK $resume &
    CHILD=$!
    wait $CHILD
    rc=$?
    CHILD=0
    echo "cell $name exit $rc"
    [ $rc -eq 0 ] || exit $rc
  done
done
python scripts/pair_context_eval.py --e-ckpt $E --ckpt-dir $CK --train-dir $TRAIN --iters $ITERS --eval-seeds 12000,12100,12200 --out results/pair_context_full/results.json
