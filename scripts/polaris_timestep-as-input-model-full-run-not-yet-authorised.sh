#!/bin/bash
#SBATCH --job-name=augur-timestep-as-input-model-full-run
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-timestep-as-input-model-full-run-%j.log
##SBATCH --dependency=afterany:SUPERVISOR_GPU_SLOT_PREDECESSOR_JOBID
# NOT YET AUTHORISED: runs only with FULL_RUN_AUTHORISED=yes in the submission environment (sbatch --export=ALL,FULL_RUN_AUTHORISED=yes). Starts per spec 10a item 9 once stage 1 does not close the experiment.
# Full run of timestep-as-input-model: variant ms_kp_pot_v_g128_dtin, 60000 iterations (the control's count), mixed step sizes, warm start from init_B, seed 4738, then eval on seeds 9100/9200/9300.
# Estimate 5 GPU-h (spec figure; untimed). Requeue-safe: full state every 500 iterations and on SIGTERM; --iters 60000 resumes from the checkpoint. Needs stage 1's truth sets in $HOME/augur_timestep_truth.
set -uo pipefail
[ "${FULL_RUN_AUTHORISED:-}" = "yes" ] || { echo "NOT AUTHORISED: stage 1 verdict and supervisor go required"; exit 64; }
export PYTHONUNBUFFERED=1
W=$HOME/augur_architecture_push_stage1_timestep
I=$HOME/augur/checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt
T=$HOME/architecture_push_stage1_code.tgz
D=$HOME/augur_timestep_truth
[ -s $T ] && [ -s $I ] || { echo "MISSING $T or $I"; exit 2; }
[ "$(ls $D/*.pt 2>/dev/null | wc -l)" -eq 15 ] || { echo "MISSING truth sets in $D"; exit 2; }
mkdir -p $W && cd $W && tar xzf $T || { echo "EXTRACT FAILED"; exit 2; }
export PYTHONPATH=$W
O=results/timestep_as_input_model_full
C=ckpt_timestep_as_input_model_full
mkdir -p $O $C
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -s $O/timestep-as-input-model_eval.json ]; then
  python scripts/train_scatter_timestep.py --mode train --init-ckpt $I --data-dir $D --out-dir $O --ckpt-dir $C --iters 60000 --seed 4738 >> $O/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "train exit $RC"; tail -20 $O/run.log; exit $RC; }
  python scripts/train_scatter_timestep.py --mode eval --data-dir $D --out-dir $O --ckpt-dir $C >> $O/run.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "eval exit $RC"; exit $RC; }
fi
[ -s $O/timestep-as-input-model_eval.json ] || { echo "MISSING OUTPUT"; exit 1; }
echo ALLDONE
