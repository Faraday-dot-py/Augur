#!/bin/bash
#SBATCH --job-name=augur-three-dimensional-sparse-voxel-scatter-field-score-slice
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=student-pleiades
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --output=augur-three-dimensional-sparse-voxel-scatter-field-score-slice-%j.log
# --dependency=afterany:TAIL_JOB_ID  (placeholder: the supervisor passes it on the sbatch command line; one GPU, counts toward the 2-GPU cap while the training segment runs)
# Scores one training slice checkpoint (score only, no training, no bench): --export=ALL,N=20000 [,SLICE_CKPT=<path>]. SLICE_CKPT defaults to the at-N copy written by the segment job.
# Writes results/three_dimensional_sparse_voxel_scatter_field-at-<N>-its-score/score.json (ratio3, ratio2, val err@20 at that iteration). Runs in its own directory so it never touches the running training tree.
# Needs $HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz and $HOME/augur_nfr/kern/E_frozen.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
N=${N:?set N to the slice iteration, e.g. N=20000}
TRAIN=$HOME/augur-three-dimensional-sparse-voxel-scatter-field
SLICE_CKPT=${SLICE_CKPT:-$TRAIN/ckpt/three_dimensional_sparse_voxel_scatter_field-at-$N-its.pt}
W=$HOME/augur-three-dimensional-sparse-voxel-scatter-field-score-$N
TGZ=$HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz
CTRL=$HOME/augur_nfr/kern/E_frozen.pt
for f in $TGZ $CTRL $SLICE_CKPT; do [ -s $f ] || { echo "MISSING $f"; exit 2; }; done
mkdir -p $W || exit 2
tar xzf $TGZ -C $W || { echo "TAR EXTRACT FAILED"; exit 2; }
ln -sfn $HOME/augur_nfr/kern $W/kern
cd $W || exit 2
export PYTHONPATH=.
for f in scripts/three_dimensional_sparse_voxel_scatter_field.py scripts/stage1_common.py scripts/stage1_preflight.py model/sparse_voxel_scatter_field.py model/__init__.py kern/E_frozen.pt; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q "ckpt-file" scripts/three_dimensional_sparse_voxel_scatter_field.py || { echo "STALE three_dimensional_sparse_voxel_scatter_field.py (no --ckpt-file)"; exit 2; }
md5sum $CTRL $SLICE_CKPT scripts/three_dimensional_sparse_voxel_scatter_field.py model/sparse_voxel_scatter_field.py
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/stage1_preflight.py --control kern/E_frozen.pt --device cuda || { echo "CONTROL PREFLIGHT FAILED"; exit 2; }
OUT=results/three_dimensional_sparse_voxel_scatter_field-at-$N-its-score
mkdir -p $OUT
python scripts/three_dimensional_sparse_voxel_scatter_field.py --mode score --control kern/E_frozen.pt --ckpt-file $SLICE_CKPT --heldout ${HELDOUT:-9100,9200,9300} --device cuda --out $OUT >> $OUT/run.log 2>&1
RC=$?
[ $RC -eq 0 ] || { echo "score exit $RC"; tail -20 $OUT/run.log; exit $RC; }
[ -s $OUT/score.json ] || { echo "MISSING OUTPUT $OUT/score.json"; exit 1; }
grep -E "ratio|val err" $OUT/run.log | tail -6
