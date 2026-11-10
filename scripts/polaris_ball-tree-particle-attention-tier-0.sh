#!/bin/bash
#SBATCH --job-name=augur-ball-tree-particle-attention-tier-0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-ball-tree-particle-attention-tier-0-%j.log
# --dependency=afterany:TAIL_JOB_ID  (placeholder: the supervisor passes it on the sbatch command line from the slot tail in docs/debugging/gpu-slot-chains.md; never submit without it)
# Stage 1 of ball-tree-particle-attention-tier-0: train 60000 its (the control's count, spec 10a.1; resumable, full-state checkpoint every 100 its), then score (err@5/20 at N=1000 cold and warm vs frozen the scatter-field control (ms_kp_pot_v_g128)) and bench (ms/tick N=1e3..1e6).
# Pre-registration: docs/debugging/architecture-push-prereg/ball-tree-particle-attention-tier-0.md. Runtime estimate (untimed): about 0.3 s/it from the nearest logged rate (width-32 scatter field, 0.28-0.31 s/it)
# gives 5 h training, plus validation, scoring and the 1e6 bench, about 5.5-6.5 GPU-h; limit 12 h per submission, requeue resumes. The 10a.1 extension is a rerun with --iters 120000 from the same checkpoint dir (needs a retrained double-length control, a user decision). Rerun with sbatch to resume (checkpoint, validation curve and bench cells are skipped when present).
# Needs $HOME/ball_tree_particle_attention_code.tgz (scripts/build_architecture_push_stage1_code_tarball.sh) and $HOME/augur_nfr/kern/E_frozen.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur-ball-tree-particle-attention
TGZ=$HOME/ball_tree_particle_attention_code.tgz
CTRL=$HOME/augur_nfr/kern/E_frozen.pt
for f in $TGZ $CTRL; do [ -s $f ] || { echo "MISSING $f"; exit 2; }; done
mkdir -p $W || exit 2
tar xzf $TGZ -C $W || { echo "TAR EXTRACT FAILED"; exit 2; }
ln -sfn $HOME/augur_nfr/kern $W/kern
cd $W || exit 2
export PYTHONPATH=.
for f in scripts/ball_tree_particle_attention.py scripts/stage1_common.py scripts/stage1_preflight.py model/ball_tree_attention.py model/__init__.py kern/E_frozen.pt; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q "it_hook" scripts/train_scatter_field.py || { echo "STALE train_scatter_field.py (no it_hook)"; exit 2; }
md5sum $CTRL scripts/scatter_field.py scripts/train_scatter_field.py scripts/ball_tree_particle_attention.py model/ball_tree_attention.py
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/stage1_preflight.py --control kern/E_frozen.pt --device cuda || { echo "CONTROL PREFLIGHT FAILED"; exit 2; }
OUT=results/ball_tree_particle_attention
mkdir -p $OUT ckpt
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/ball_tree_particle_attention.py --mode all --control kern/E_frozen.pt --iters ${ITERS:-60000} --heldout ${HELDOUT:-9100,9200,9300} --device cuda --out $OUT --ckpt-dir ckpt >> $OUT/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "ball_tree_particle_attention exit $RC"; tail -20 $OUT/run.log; exit $RC; }
for f in train_done.json val_curve.json score.json bench.json summary.json; do
  [ -s $OUT/$f ] || { echo "MISSING OUTPUT $OUT/$f"; exit 1; }
done
cat $OUT/summary.json
