#!/bin/bash
#SBATCH --job-name=augur-three-dimensional-sparse-voxel-scatter-field
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=student-pleiades
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-three-dimensional-sparse-voxel-scatter-field-%j.log
# --dependency=afterany:TAIL_JOB_ID  (placeholder: the supervisor passes it on the sbatch command line from the slot tail in docs/debugging/gpu-slot-chains.md; never submit without it)
# Stage 1 of three-dimensional-sparse-voxel-scatter-field (spec cost 1.5 GPU-h): train ITERS its (default 4000, resumable, full-state checkpoint every 100 its), then score (cold and virial Plummer spheres vs exact 3D truth,
# 2D the scatter-field control (ms_kp_pot_v_g128) control at equal N) and bench (cost per body-tick, 3D model vs 2D control). The 20 GPU-h full run uses this same script with ITERS and --batch raised and needs the user's go-ahead after the stage 1 report:
# sbatch --export=ALL,ITERS=60000,BATCH=16,KEND=30 --time=<longer> this.sh; every requeue resumes from the checkpoint.
# Pre-registration: docs/debugging/architecture-push-prereg/three-dimensional-sparse-voxel-scatter-field.md. Runtime estimate (untimed, no 3D rate is logged): the nearest logged rate is 0.28-0.31 s/it for the 2D width-32 model;
# assuming up to 2x for sparse 3D at batch 8 gives 4000 its in about 40 min, plus validation, scoring and bench: 1.0-1.5 GPU-h; limit 4 h.
# Needs $HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz (scripts/build_architecture_push_stage1_code_tarball.sh) and $HOME/augur_nfr/kern/E_frozen.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
ITERS=${ITERS:-4000}
BATCH=${BATCH:-8}
KEND=${KEND:-20}
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
md5sum $CTRL scripts/scatter_field.py scripts/train_scatter_field.py scripts/three_dimensional_sparse_voxel_scatter_field.py model/sparse_voxel_scatter_field.py
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/stage1_preflight.py --control kern/E_frozen.pt --device cuda || { echo "CONTROL PREFLIGHT FAILED"; exit 2; }
OUT=results/three_dimensional_sparse_voxel_scatter_field
mkdir -p $OUT ckpt
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/three_dimensional_sparse_voxel_scatter_field.py --mode all --control kern/E_frozen.pt --iters $ITERS --heldout ${HELDOUT:-9100,9200,9300} --batch $BATCH --k-end $KEND --device cuda --out $OUT --ckpt-dir ckpt >> $OUT/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "three_dimensional_sparse_voxel_scatter_field exit $RC"; tail -20 $OUT/run.log; exit $RC; }
for f in train_done.json val_curve.json score.json bench.json summary.json; do
  [ -s $OUT/$f ] || { echo "MISSING OUTPUT $OUT/$f"; exit 1; }
done
cat $OUT/summary.json
