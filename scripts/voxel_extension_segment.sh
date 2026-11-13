#!/bin/bash
#SBATCH --job-name=augur-three-dimensional-sparse-voxel-scatter-field
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=student-pleiades
#SBATCH --time=09:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-three-dimensional-sparse-voxel-scatter-field-extension-segment-%j.log
# --dependency=afterany:TAIL_JOB_ID  (placeholder: the supervisor passes it on the sbatch command line from the slot tail in docs/debugging/gpu-slot-chains.md; never submit without it)
# One segment of the 60000-iteration voxel extension: resumes ckpt/three_dimensional_sparse_voxel_scatter_field.pt and trains to STOP_AT (env, e.g. 20000), then saves, copies the checkpoint to
# ckpt/three_dimensional_sparse_voxel_scatter_field-at-<STOP_AT>-its.pt and exits 0 without train_done.json. The cosine lr schedule stays a function of ITERS (default 60000), so segments reproduce one uninterrupted run.
# STOP_AT unset, 0 or equal to ITERS: final segment, trains to ITERS then runs score and bench (mode all) and needs all outputs.
# Sizing: --export=ALL,STOP_AT=20000 and --time proportional to (STOP_AT - checkpoint it) times the logged s_per_it_first100 plus validation.
# Slice scoring runs separately and concurrently (scripts/voxel_score_slice.sh); it counts toward the 2-GPU cap.
# Needs $HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz (scripts/build_architecture_push_stage1_code_tarball.sh) and $HOME/augur_nfr/kern/E_frozen.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
ITERS=${ITERS:-60000}
STOP_AT=${STOP_AT:-0}
BATCH=${BATCH:-8}
KEND=${KEND:-20}
[ "$STOP_AT" -le "$ITERS" ] || { echo "STOP_AT $STOP_AT exceeds ITERS $ITERS"; exit 2; }
W=$HOME/augur-three-dimensional-sparse-voxel-scatter-field
TGZ=$HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz
CTRL=$HOME/augur_nfr/kern/E_frozen.pt
for f in $TGZ $CTRL; do [ -s $f ] || { echo "MISSING $f"; exit 2; }; done
mkdir -p $W || exit 2
tar xzf $TGZ -C $W || { echo "TAR EXTRACT FAILED"; exit 2; }
ln -sfn $HOME/augur_nfr/kern $W/kern
cd $W || exit 2
export PYTHONPATH=.
for f in scripts/three_dimensional_sparse_voxel_scatter_field.py scripts/stage1_common.py scripts/stage1_preflight.py model/sparse_voxel_scatter_field.py model/__init__.py kern/E_frozen.pt; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q "it_hook" scripts/train_scatter_field.py || { echo "STALE train_scatter_field.py (no it_hook)"; exit 2; }
grep -q "exact_resume" scripts/train_scatter_field.py || { echo "STALE train_scatter_field.py (no exact_resume)"; exit 2; }
grep -q "stop-at" scripts/three_dimensional_sparse_voxel_scatter_field.py || { echo "STALE three_dimensional_sparse_voxel_scatter_field.py (no --stop-at)"; exit 2; }
md5sum $CTRL scripts/scatter_field.py scripts/train_scatter_field.py scripts/three_dimensional_sparse_voxel_scatter_field.py model/sparse_voxel_scatter_field.py
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/stage1_preflight.py --control kern/E_frozen.pt --device cuda || { echo "CONTROL PREFLIGHT FAILED"; exit 2; }
OUT=results/three_dimensional_sparse_voxel_scatter_field
CK=ckpt/three_dimensional_sparse_voxel_scatter_field
mkdir -p $OUT ckpt
if [ -s $OUT/train_done.json ]; then
  [ -d $OUT-at-10000-its ] || cp -r $OUT $OUT-at-10000-its
  [ -e $CK-at-10000-its.pt ] || cp $CK.pt $CK-at-10000-its.pt
  rm -f $OUT/train_done.json
fi
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
MODE=train
[ "$STOP_AT" -gt 0 ] && [ "$STOP_AT" -lt "$ITERS" ] || MODE=all
python scripts/three_dimensional_sparse_voxel_scatter_field.py --mode $MODE --control kern/E_frozen.pt --iters $ITERS --stop-at $STOP_AT --heldout ${HELDOUT:-9100,9200,9300} --batch $BATCH --k-end $KEND --device cuda --out $OUT --ckpt-dir ckpt >> $OUT/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "three_dimensional_sparse_voxel_scatter_field exit $RC"; tail -20 $OUT/run.log; exit $RC; }
if [ "$MODE" = train ]; then
  [ -s $CK-at-$STOP_AT-its.pt ] || { echo "MISSING slice checkpoint $CK-at-$STOP_AT-its.pt"; exit 1; }
  [ ! -e $OUT/train_done.json ] || { echo "train_done.json written at a stop-at segment"; exit 1; }
  grep "STOP-AT $STOP_AT" $OUT/run.log | tail -1
  exit 0
fi
for f in train_done.json val_curve.json score.json bench.json summary.json; do
  [ -s $OUT/$f ] || { echo "MISSING OUTPUT $OUT/$f"; exit 1; }
done
cat $OUT/summary.json
