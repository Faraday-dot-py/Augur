#!/bin/bash
#SBATCH --job-name=augur-neuron-graph-extra-training-seeds-bundle
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --nice=100
#SBATCH --output=augur-neuron-graph-extra-training-seeds-bundle-%j.log

# training seeds 4739 and 4740 (env SEEDS) of: slow-patch-rate-token-with-patch-count-head, windowed-causal-temporal-attention-memory, plastic-synapse-state STDP probe, bit-output-cell-mlp; slow-patch and windowed each against a baseline re-trained at the same seed.
# Resumable: every trainer saves each finished arm's weights and skips it on rerun, finished steps are skipped by their result file, SIGTERM stops the current step and requeues. No single step runs over 15 min, so there is no mid-arm checkpoint.
# needs ~/neuron_extra_training_seeds_code.tgz (model/ scripts/ tests/ conftest.py).
export PYTHONUNBUFFERED=1
DEADLINE=1794070800
REMAIN=$((DEADLINE-$(date +%s)))
[ $REMAIN -le 300 ] && { echo "DEADLINE 2026-11-07 09:00 PST: $REMAIN s left, not starting"; exit 3; }
past_deadline() { [ $((DEADLINE-$(date +%s))) -le 300 ]; }
dlbg() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; exit 124; }
  exec timeout -k 120 -s TERM $left "$@"
}
dlrun() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; return 124; }
  timeout -k 120 -s TERM $left "$@"
}
D="$HOME/augur-neuron-extra-training-seeds"
TGZ="$HOME/neuron_extra_training_seeds_code.tgz"
test -f $TGZ || { echo MISSING $TGZ; exit 2; }
mkdir -p "$D" && tar -xzf $TGZ -C "$D"
cd "$D" || exit 2
export PYTHONPATH=.
for f in model/neuron_graph.py model/neuron_graph_plastic.py model/neuron_graph_slow_patch.py model/neuron_graph_windowed_attention.py model/neuron_graph_variants.py scripts/neuron_circuit_teacher.py scripts/neuron_plasticity_teacher.py scripts/neuron_loss_diagnostic.py scripts/neuron_spike_tick_investigation.py scripts/train_neuron_graph_smoke.py scripts/train_neuron_graph_loss_variants.py scripts/train_neuron_graph_baseline_seed.py scripts/train_neuron_graph_slow_patch.py scripts/train_neuron_graph_windowed_attention.py scripts/train_neuron_plasticity_smoke.py scripts/bit_force_smoke.py scripts/run_without_native_triton_ops.py scripts/score_neuron_extra_training_seeds.py tests/test_neuron_graph.py tests/test_neuron_graph_slow_patch.py tests/test_neuron_graph_windowed_attention.py tests/test_neuron_plasticity.py conftest.py; do test -f $f || { echo MISSING $f; exit 2; }; done
R=results/neuron_extra_training_seeds
SEEDS=${SEEDS:-"4739 4740"}
mkdir -p $R/bit-output
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "restart count ${SLURM_RESTART_COUNT:-0}"
python -m pytest tests/test_neuron_graph.py tests/test_neuron_graph_slow_patch.py tests/test_neuron_graph_windowed_attention.py tests/test_neuron_plasticity.py -x -q || exit 1

FAILED=""
TERMED=""
CHILD=""
trap 'echo "[$(date -Iseconds)] TERM: stopping current step, requeue"; TERMED=1; [ -n "$CHILD" ] && kill -TERM $CHILD 2>/dev/null' TERM

step() {
  NAME=$1; DONE=$2; shift 2
  [ -n "$TERMED" ] && return
  if [ -e $DONE ]; then echo "STEP $NAME skipped, $DONE exists"; return; fi
  if past_deadline; then echo "STEP $NAME skipped, deadline"; DEADLINED=1; return; fi
  dlbg "$@" &
  CHILD=$!
  while kill -0 $CHILD 2>/dev/null; do wait $CHILD; done
  wait $CHILD
  RC=$?
  CHILD=""
  echo "STEP $NAME exit code $RC"
  if past_deadline && [ ! -e $DONE ]; then DEADLINED=1; return; fi
  if [ $RC -ne 0 ] && [ -z "$TERMED" ]; then FAILED="$FAILED $NAME"; fi
}

needs_baseline() {
  B=$R/baseline/seed$1/full-synaptic-messages.pt
  [ -e $B ] && return 0
  [ -n "$TERMED" ] && return 1
  echo "STEP $2 exit code 3 (baseline $B missing)"; FAILED="$FAILED $2"; return 1
}

for S in $SEEDS; do
  step baseline-seed$S $R/baseline/seed$S/full-synaptic-messages.pt python scripts/train_neuron_graph_baseline_seed.py --seed $S --out-dir $R/baseline
done
for S in $SEEDS; do
  needs_baseline $S slow-patch-seed$S && step slow-patch-seed$S $R/slow-patch/seed$S/results.json python scripts/train_neuron_graph_slow_patch.py --seed $S --baseline-dir $R/baseline/seed$S --out-dir $R/slow-patch
done
for S in $SEEDS; do
  needs_baseline $S windowed-attention-seed$S && step windowed-attention-seed$S $R/windowed-attention/seed$S/results.json python scripts/run_without_native_triton_ops.py scripts/train_neuron_graph_windowed_attention.py --seed $S --baseline-dir $R/baseline/seed$S --out-dir $R/windowed-attention
done
for S in $SEEDS; do
  step plasticity-seed$S $R/plasticity/seed$S/results.json python scripts/train_neuron_plasticity_smoke.py --seed $S --out-dir $R/plasticity
done
for S in $SEEDS; do
  step bit-output-seed$S $R/bit-output/seed$S.json python scripts/bit_force_smoke.py --seed $S --out $R/bit-output/seed$S.json
done

if [ -n "${DEADLINED:-}" ] || past_deadline; then
  echo "DEADLINE_STOP: unfinished steps skipped, scoring what exists"
  python scripts/score_neuron_extra_training_seeds.py --root $R
  exit 4
fi
if [ -n "$TERMED" ]; then
  echo "unfinished, requeue"; scontrol requeue $SLURM_JOB_ID; exit 0
fi
python scripts/score_neuron_extra_training_seeds.py --root $R
SC=$?
echo "STEP score exit code $SC"
[ $SC -ne 0 ] && FAILED="$FAILED score"
if [ -n "$FAILED" ]; then echo "FAILED STEPS:$FAILED"; exit 1; fi
echo ALLDONE
