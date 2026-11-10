#!/bin/bash
#SBATCH --job-name=augur-staggered-double-grid-with-quadratic-deposit-full
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=09:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-staggered-double-grid-with-quadratic-deposit-full-%j.log
# NOT YET AUTHORISED. Do not sbatch until the supervisor has read stage 1 (shift_average_stage1 stage1.json; closure rule in docs/debugging/architecture-push-prereg/staggered-double-grid-with-quadratic-deposit.md)
# and created $HOME/augur_staggered_double_grid/FULL_RUN_AUTHORISED (the script exits 3 without it). One GPU only; the supervisor owns the slot chain.
# Trains ms_kp_pot_v_g128_quad_dual (quadratic three-cell deposit and gather, two grids offset half a cell, same U-Net) for the control's 60769 iterations with the the frozen single-law gravity checkpoint recipe (init_B warm start,
# lr 5e-4, batch 16, k 4-30, 1500 scenes x 50 steps, seed 4738), validation loss every 500 iterations for the plateau guard, then evaluates candidate vs frozen the frozen single-law gravity checkpoint on seeds 9100/9200/9300 and
# 12000/12100/12200 and writes verdicts. Requeue-safe on preemption or node failure (a 9 h timeout needs a manual resubmit, which resumes): the trainer saves full state (model, optimiser, iteration) every 100 iterations and on SIGTERM, and resumes from it; finished steps are skipped.
# Estimate 5-6 GPU-h (UNVERIFIED: the frozen single-law gravity checkpoint took 9000 s for 60769 iterations; two grids roughly double it) plus 0.5 evals. Spec cost 3 GPU-h; see prereg. Code tarball:
# bash scripts/build_architecture_push_code_tarball.sh staggered_double_grid_code.tgz, upload to $HOME.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_staggered_double_grid
mkdir -p $W
[ -e $W/FULL_RUN_AUTHORISED ] || { echo "NOT AUTHORISED: create $W/FULL_RUN_AUTHORISED after reading stage 1"; exit 3; }
cd $W && tar xzf $HOME/staggered_double_grid_code.tgz
export PYTHONPATH=$W
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur/checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt
N=single-law-gravity-checkpoint-g128
V=ms_kp_pot_v_g128_quad_dual
ITERS=60769
CK=$W/ckpt/${N}_$V.pt
O=$W/results
mkdir -p $W/ckpt $O
for f in $E $INIT scripts/train_scatter_field.py scripts/staggered_double_grid_eval.py scripts/staggered_double_grid_verdict.py scripts/plateau_guard.py scripts/scatter_field.py model/__init__.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q 'ms_kp_pot_v_g128_quad_dual' scripts/train_scatter_field.py || { echo "MISSING variant in train_scatter_field.py"; exit 2; }
nvidia-smi -L || { echo "NO GPU"; exit 2; }

FAIL=0
CHILD=
trap 'echo SIGTERM; [ -n "$CHILD" ] && kill -TERM $CHILD && wait $CHILD; exit 143' TERM
step() {
  name=$1; shift
  "$@" &
  CHILD=$!
  wait $CHILD
  rc=$?
  CHILD=
  echo STEP $name $rc
  [ $rc -eq 0 ] || FAIL=1
  return $rc
}

if [ ! -e $O/TRAIN_DONE ]; then
  if step train python scripts/train_scatter_field.py --exp $N --variant $V --train 1500 --steps 50 --batch 16 --k-start 4 --k-end 30 --lr 5e-4 --iters $ITERS \
    --eval-steps 21 --eval-scenes 2 --log-every 100 --val-every 500 --val-scenes 32 --save-on-sigterm --device cuda --seed 4738 --init-ckpt $INIT --out-dir $O/trainer --ckpt-dir $W/ckpt; then touch $O/TRAIN_DONE; fi
fi
[ -e $O/TRAIN_DONE ] || { echo "TRAIN NOT DONE (iterations short of $ITERS or step failed)"; exit 1; }

step plateau_guard bash -c "set -o pipefail; python scripts/plateau_guard.py $CK.val.jsonl $ITERS | tee $O/plateau_guard.json"
for S in 9100,9200,9300:primary 12000,12100,12200:fresh; do
  SEEDS=${S%%:*}; TAG=${S##*:}
  [ -e $O/eval_$TAG.json ] || step eval_$TAG python scripts/staggered_double_grid_eval.py --control $E --cand $CK --variant $V --seeds $SEEDS --out $O/eval_$TAG.json
  [ -s $O/eval_$TAG.json ] && step verdict_$TAG bash -c "set -o pipefail; python scripts/staggered_double_grid_verdict.py full $O/eval_$TAG.json --guard '$(cat $O/plateau_guard.json)' | tee $O/verdict_$TAG.json"
done
for f in $CK $CK.val.jsonl $O/plateau_guard.json $O/eval_primary.json $O/eval_fresh.json $O/verdict_primary.json $O/verdict_fresh.json; do [ -s $f ] || { echo OUTPUT MISSING $f; FAIL=1; }; done
echo "FAIL=$FAIL"
exit $FAIL
