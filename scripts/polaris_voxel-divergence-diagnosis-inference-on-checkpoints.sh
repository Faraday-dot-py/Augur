#!/bin/bash
#SBATCH --job-name=augur-voxel-divergence-diagnosis-inference-on-checkpoints
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --dependency=afterany:11101
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-voxel-divergence-diagnosis-inference-on-checkpoints-%j.log
# --dependency=afterany:<slot 2 tail job id> is passed on the sbatch command line (never submit without it).
# Inference only: weights, train-vs-val gap, tick-0 force parts with ablations, size sweep over the five stage-1/extension checkpoints. Resumes per cell (sweep.json, components.json) after a requeue.
# Needs $HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz, $HOME/voxel-divergence-diagnosis/ckpt/{voxel-it10000,voxel-it15000,voxel-it20000,voxel-it25000,voxel-at-30000-its}.pt and
# $HOME/voxel-divergence-diagnosis/voxel_divergence_diagnosis.py.
set -uo pipefail
export PYTHONUNBUFFERED=1
D=$HOME/voxel-divergence-diagnosis
W=$D/work
TGZ=$HOME/three_dimensional_sparse_voxel_scatter_field_code.tgz
for f in $TGZ $D/voxel_divergence_diagnosis.py $D/ckpt/voxel-it10000.pt $D/ckpt/voxel-it15000.pt $D/ckpt/voxel-it20000.pt $D/ckpt/voxel-it25000.pt $D/ckpt/voxel-at-30000-its.pt; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
mkdir -p $W || exit 2
tar xzf $TGZ -C $W || { echo "TAR EXTRACT FAILED"; exit 2; }
cp $D/voxel_divergence_diagnosis.py $W/scripts/ || exit 2
cd $W || exit 2
export PYTHONPATH=.
for f in scripts/three_dimensional_sparse_voxel_scatter_field.py scripts/scatter_field.py scripts/scatter_scaling_ops.py scripts/stage1_common.py model/sparse_voxel_scatter_field.py model/__init__.py scripts/voxel_divergence_diagnosis.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
md5sum $D/ckpt/*.pt scripts/voxel_divergence_diagnosis.py model/sparse_voxel_scatter_field.py scripts/three_dimensional_sparse_voxel_scatter_field.py
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$D/results
mkdir -p $OUT
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/voxel_divergence_diagnosis.py --mode all --ckpt-dir $D/ckpt --out $OUT --device cuda >> $OUT/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "voxel_divergence_diagnosis exit $RC"; tail -20 $OUT/run.log; exit $RC; }
for f in weights.json gap.json components.json sweep.json; do
  [ -s $OUT/$f ] || { echo "MISSING OUTPUT $OUT/$f"; exit 1; }
done
tail -3 $OUT/run.log
