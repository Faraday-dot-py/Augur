#!/bin/bash
#SBATCH --job-name=augur-one-network-for-all-tiers-with-scale-input
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SUPERVISOR_REPLACES_WITH_SLOT_TAIL_JOB_ID
#SBATCH --output=augur-one-network-for-all-tiers-with-scale-input-%j.log
# Both arms trained jointly on the far-field target at span 256 (grid 512, 4 tiers), same budget, seed 4738: shared weights with a scale input, and per-tier weights (control).
# Then far-field evaluation on seeds 9100/9200/9300: shared at span 256 (4 tiers) and span 1024 (6 tiers, two unseen), control at span 256. Pre-registration: docs/debugging/architecture-push-prereg/one-network-for-all-tiers-with-scale-input.md.
# Estimate 6 GPU-h (spec 1 / 6): no logged rate for far-target joint training at span 256; this is the spec figure, untested. Full-state checkpoint every 250 its, --resume, SIGTERM saves and exits 143; sbatch again to resume.
# Needs $HOME/augur-architecture-push-stage-one-code.tgz. No other input checkpoint (the far target has a zero tier 0).
set -uo pipefail
export PYTHONUNBUFFERED=1
CODE=$HOME/augur-architecture-push-stage-one-code.tgz
W=$HOME/augur-one-network-for-all-tiers-with-scale-input
[ -s $CODE ] || { echo "MISSING $CODE"; exit 2; }
mkdir -p $W/ckpt $W/results || exit 2
SUM=$(md5sum $CODE | cut -d' ' -f1)
[ -e $W/.extracted-$SUM ] || { tar xzf $CODE -C $W && touch $W/.extracted-$SUM; } || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=$W
for f in scripts/train_tiered.py scripts/score_architecture_push_stage_one.py model/tiered/tier.py model/tiered/hierarchy.py; do
  [ -s $f ] || { echo "MISSING $f in tarball"; exit 2; }
done
grep -q -- "--shared-scale" scripts/train_tiered.py || { echo "STALE train_tiered.py: no --shared-scale"; exit 2; }
grep -q "scale_ch" model/tiered/tier.py || { echo "STALE tier.py: no scale_ch"; exit 2; }
md5sum scripts/train_tiered.py model/tiered/tier.py model/tiered/hierarchy.py
nvidia-smi --query-gpu=name,memory.total --format=csv
ITERS=3000
COMMON="--target far --joint --tiers 4 --span 256 --width 32 --layers 6 --far-radius 20 --iters $ITERS --seed 4738 --ckpt-dir ckpt --ckpt-every 250"
SH=shared-scale-input-s4738
CT=per-tier-weights-control-s4738
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
for tag in $SH $CT; do
  [ -e ckpt/$tag.trained ] && continue
  if [ $tag = $SH ]; then X="--shared-scale --val-every 100"; else X="--val-every 100"; fi
  python scripts/train_tiered.py --mode S --tag $tag $COMMON $X --resume >> $tag.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "train $tag exit $RC"; tail -20 $tag.log; exit $RC; }
  grep -q "S done" $tag.log || { echo "$tag did not finish (no 'S done')"; exit 1; }
  [ -s ckpt/tiered_S$tag.pt ] || { echo "MISSING ckpt/tiered_S$tag.pt"; exit 1; }
  touch ckpt/$tag.trained
done
PID=
ev() {
  [ -s results/$1.json ] && return 0
  python scripts/train_tiered.py --mode far --tag $2 --target far --tiers $3 --span $4 --width 32 --layers 6 --far-radius 20 --ckpt-dir ckpt --eval-batch 2 $5 --out results/$1.json >> $2.eval.log 2>&1 || { echo "eval $1 failed"; tail -20 $2.eval.log; exit 1; }
  [ -s results/$1.json ] || { echo "MISSING results/$1.json"; exit 1; }
}
ev shared_256 $SH 4 256 --shared-scale
ev shared_1024 $SH 6 1024 --shared-scale
ev control_256 $CT 4 256 ""
python scripts/score_architecture_push_stage_one.py scale-input --in shared_256=results/shared_256.json --in shared_1024=results/shared_1024.json --in control_256=results/control_256.json --val ckpt/tiered_S$SH.pt --iters $ITERS --out results/one-network-for-all-tiers-with-scale-input-score.json || exit 1
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
