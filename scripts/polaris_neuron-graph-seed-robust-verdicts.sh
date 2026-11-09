#!/bin/bash
#SBATCH --job-name=augur-neuron-graph-seed-robust-verdicts
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --nice=100
#SBATCH --dependency=afterany:7192
#SBATCH --output=augur-neuron-graph-seed-robust-verdicts-%j.log

# seeds 4738-4747 (env SEEDS) of slow-patch-rate-token-with-patch-count-head (baseline and 48-tick control re-trained per seed) and the plastic-synapse-state STDP probe (fixed-synapse arm trained per seed), original pass/kill lines, then the aggregate scorer.
# Resumable: every trainer saves each finished arm's weights and skips it on rerun, finished steps are skipped by their results file, SIGTERM stops the current step and requeues.
# needs ~/neuron_seed_robust_code.tgz (model/ scripts/ tests/ conftest.py).
export PYTHONUNBUFFERED=1
D="$HOME/augur-neuron-seed-robust"
TGZ="$HOME/neuron_seed_robust_code.tgz"
test -f $TGZ || { echo MISSING $TGZ; exit 2; }
mkdir -p "$D" && tar -xzf $TGZ -C "$D"
cd "$D" || exit 2
export PYTHONPATH=.
for f in model/neuron_graph.py model/neuron_graph_plastic.py model/neuron_graph_slow_patch.py scripts/neuron_plasticity_teacher.py scripts/train_neuron_graph_smoke.py scripts/train_neuron_graph_loss_variants.py scripts/train_neuron_graph_baseline_seed.py scripts/train_neuron_graph_slow_patch.py scripts/train_neuron_plasticity_smoke.py scripts/score_neuron_seed_robust_verdicts.py tests/test_neuron_graph.py tests/test_neuron_graph_slow_patch.py tests/test_neuron_plasticity.py conftest.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results/neuron_seed_robust
SEEDS=${SEEDS:-"4738 4739 4740 4741 4742 4743 4744 4745 4746 4747"}
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "restart count ${SLURM_RESTART_COUNT:-0}"
python -m pytest tests/test_neuron_graph.py tests/test_neuron_graph_slow_patch.py tests/test_neuron_plasticity.py -x -q || exit 1

FAILED=""
TERMED=""
CHILD=""
trap 'echo "[$(date -Iseconds)] TERM: stopping current step, requeue"; TERMED=1; [ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null' TERM

step() {
  NAME=$1; DONE=$2; shift 2
  [ -n "$TERMED" ] && return
  if [ -e $DONE ]; then echo "STEP $NAME skipped, $DONE exists"; return; fi
  "$@" &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do wait $CHILD; done
  wait $CHILD
  RC=$?
  CHILD=""
  echo "STEP $NAME exit code $RC"
  if [ $RC -ne 0 ] && [ -z "$TERMED" ]; then FAILED="$FAILED $NAME"; fi
}

for S in $SEEDS; do
  step baseline-seed$S $R/baseline/seed$S/full-synaptic-messages.pt python scripts/train_neuron_graph_baseline_seed.py --seed $S --out-dir $R/baseline
  if [ -e $R/baseline/seed$S/full-synaptic-messages.pt ]; then
    step slow-patch-seed$S $R/slow-patch/seed$S/results.json python scripts/train_neuron_graph_slow_patch.py --seed $S --baseline-dir $R/baseline/seed$S --out-dir $R/slow-patch
  elif [ -z "$TERMED" ]; then
    echo "STEP slow-patch-seed$S exit code 3 (baseline missing)"; FAILED="$FAILED slow-patch-seed$S"
  fi
  step plasticity-seed$S $R/plasticity/seed$S/results.json python scripts/train_neuron_plasticity_smoke.py --seed $S --out-dir $R/plasticity
done

if [ -n "$TERMED" ]; then
  echo "unfinished, requeue"; scontrol requeue $SLURM_JOB_ID; exit 0
fi
python scripts/score_neuron_seed_robust_verdicts.py --root $R
SC=$?
echo "STEP score exit code $SC"
[ $SC -ne 0 ] && FAILED="$FAILED score"
if [ -n "$FAILED" ]; then echo "FAILED STEPS:$FAILED"; exit 1; fi
echo ALLDONE
