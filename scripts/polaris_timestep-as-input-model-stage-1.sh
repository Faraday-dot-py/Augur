#!/bin/bash
#SBATCH --job-name=augur-timestep-as-input-model-stage-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-timestep-as-input-model-stage-1-%j.log
# Stage 1 of timestep-as-input-model: exact truth at step sizes 0.05/0.1/0.2 (train + eval) and 0.075 (eval only) on the Exp E scene distribution, then the frozen Exp E reference eval at each step size.
# Pre-registration: docs/debugging/architecture-push-prereg/timestep-as-input-model.md.
# Estimate about 1.0 GPU-h (untimed; spec figure, 4500 train scenes + 576 eval scenes of float64 truth, then 4 x 3 frozen evals). Requeue-safe: every set is written atomically and skipped on rerun.
# Code tarball: bash scripts/build_architecture_push_stage1_code_tarball.sh, upload to $HOME. Supervisor adds --dependency at submission.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_architecture_push_stage1_timestep
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
T=$HOME/architecture_push_stage1_code.tgz
[ -s $T ] || { echo "MISSING $T"; exit 2; }
[ -s $E ] || { echo "MISSING Exp E checkpoint $E"; exit 2; }
mkdir -p $W && cd $W && tar xzf $T || { echo "EXTRACT FAILED"; exit 2; }
export PYTHONPATH=$W
for f in scripts/train_scatter_timestep.py scripts/scatter_field.py scripts/train_scatter_field.py scripts/gravity_sim.py; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q "dtin" scripts/scatter_field.py || { echo "STALE scatter_field.py (no dtin)"; exit 2; }
md5sum scripts/scatter_field.py scripts/train_scatter_timestep.py $E
nvidia-smi --query-gpu=name,memory.total --format=csv
D=$HOME/augur_timestep_truth
O=results/timestep_as_input_model_stage1
mkdir -p $D $O
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/train_scatter_timestep.py --mode gen --data-dir $D --out-dir $O --seed 4738 >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "gen exit $RC"; tail -20 $O/run.log; exit $RC; }
n=$(ls $D/*.pt | wc -l)
[ "$n" -eq 15 ] || { echo "EXPECTED 15 truth sets (3 train + 12 eval), found $n"; ls $D; exit 1; }
if [ ! -s $O/frozen_reference.json ]; then
  python scripts/train_scatter_timestep.py --mode frozen --init-ckpt $E --data-dir $D --out-dir $O >> $O/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "frozen exit $RC"; tail -20 $O/run.log; exit $RC; }
fi
[ -s $O/frozen_reference.json ] || { echo "MISSING OUTPUT $O/frozen_reference.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
