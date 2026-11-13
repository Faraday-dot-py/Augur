#!/bin/bash
#SBATCH --job-name=augur-neighbour-context-pair-branch-train-seed-replication
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --nice=100
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SLOT_TAIL_PLACEHOLDER
#SBATCH --output=augur-neighbour-context-pair-branch-train-seed-replication-%j.log
# Train seeds 4738 and 4740 x (matched control fine-tune, candidate fine-tune) from the base checkpoint (ms_kp_pot_v_g128), 1500 its each, then scoring on held-out 9100-9300 and fresh 12000-12200. Est 0.5 GPU-h, 1 GPU.
# Pre-registration: docs/debugging/architecture-push-prereg/neighbour-context-pair-branch-train-seed-replication.md
set -uo pipefail
export PYTHONUNBUFFERED=1
TGZ=${TGZ:-$HOME/augur-neighbour-context-pair-branch-train-seed-replication-code.tgz}
W=$HOME/augur_pair_context_replication
test -f $TGZ || { echo MISSING $TGZ; exit 2; }
mkdir -p $W && tar xzf $TGZ -C $W || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=.
E=$HOME/E_ms_kp_pot_v_g128.pt
ITERS=1500
SCENES=1000
TRAIN=results/pair_context_replication/train
CK=ckpt
for f in $E scripts/train_scatter_ft.py scripts/pair_context_eval.py scripts/scatter_field.py scripts/scatter_ft_data.py scripts/train_scatter_field.py; do test -f $f || { echo MISSING $f; exit 2; }; done
grep -q pairctx scripts/scatter_field.py || { echo STALE scatter_field.py without pairctx; exit 2; }
grep -q seed-replication scripts/pair_context_eval.py || { echo STALE pair_context_eval.py without seed-replication lines; exit 2; }
mkdir -p $TRAIN $CK
CHILD=0
trap '[ $CHILD -ne 0 ] && kill -TERM $CHILD 2>/dev/null; wait; echo SIGTERM handled; exit 143' TERM
nvidia-smi --query-gpu=name,memory.total --format=csv
for seed in 4738 4740; do
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
    test -f $TRAIN/${name}_train.json || { echo "MISSING output for $name"; exit 3; }
  done
done
python scripts/pair_context_eval.py --e-ckpt $E --ckpt-dir $CK --train-dir $TRAIN --train-seeds 4738,4740 --iters $ITERS --lines seed-replication --out results/pair_context_replication/results.json
rc=$?
echo "eval held-out exit $rc"
test -f results/pair_context_replication/results.json || { echo "MISSING results.json"; exit 3; }
python scripts/pair_context_eval.py --e-ckpt $E --ckpt-dir $CK --train-dir $TRAIN --train-seeds 4738,4740 --iters $ITERS --eval-seeds 12000,12100,12200 --lines seed-replication --out results/pair_context_replication/results_fresh.json
rc2=$?
echo "eval fresh exit $rc2"
test -f results/pair_context_replication/results_fresh.json || { echo "MISSING results_fresh.json"; exit 3; }
[ $rc -eq 0 ] && [ $rc2 -eq 0 ]
