#!/bin/bash
#SBATCH --job-name=augur-neuron-graph-seed-reproduction-11-07-configuration
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120
#SBATCH --nice=100
#SBATCH --dependency=afterany:7298
#SBATCH --output=augur-neuron-graph-seed-reproduction-11-07-configuration-%j.log

# job 6163 (11-07) configuration of the slow-patch rate token: byte-identical code tarball ~/neuron_extra_training_seeds_code.tgz, baseline re-trained per seed, default flags, seeds 4738 and 4739, repeated 3 times (env REPS) in separate dirs to measure run-to-run variation at fixed seed.
# Resumable: finished steps are skipped by their result file, trainers skip finished arms, SIGTERM stops the current step and requeues.
# Add --dependency=afterany:<slot tail> at submit time (gpu-slot-chains.md).
export PYTHONUNBUFFERED=1
D="$HOME/augur-neuron-seed-reproduction-11-07-configuration"
TGZ="$HOME/neuron_extra_training_seeds_code.tgz"
test -f $TGZ || { echo MISSING $TGZ; exit 2; }
mkdir -p "$D" && tar -xzf $TGZ -C "$D"
cd "$D" || exit 2
test -f $HOME/score_neuron_seed_reproduction.py || { echo MISSING $HOME/score_neuron_seed_reproduction.py; exit 2; }
cp $HOME/score_neuron_seed_reproduction.py scripts/
export PYTHONPATH=.
for f in model/neuron_graph.py model/neuron_graph_slow_patch.py scripts/neuron_circuit_teacher.py scripts/train_neuron_graph_smoke.py scripts/train_neuron_graph_loss_variants.py scripts/train_neuron_graph_baseline_seed.py scripts/train_neuron_graph_slow_patch.py scripts/score_neuron_seed_reproduction.py tests/test_neuron_graph.py tests/test_neuron_graph_slow_patch.py conftest.py; do test -f $f || { echo MISSING $f; exit 2; }; done
cat > md5.expected <<EOF
356a75b0adb67ebe8b6d7c9561a56b6c  model/neuron_graph.py
239bd420e643cebe4aacccc6d4b1176c  model/neuron_graph_slow_patch.py
6464842278de7a180ead37ca2f7110af  scripts/neuron_circuit_teacher.py
01f801b8929e267469da4b198027aba2  scripts/train_neuron_graph_smoke.py
d4d389a1a1aa497883d0a47923b17d5a  scripts/train_neuron_graph_loss_variants.py
c716c30f07480d5493eaed79859cc966  scripts/train_neuron_graph_baseline_seed.py
e821f0e9a8ac1c3ebc793e3169647eb5  scripts/train_neuron_graph_slow_patch.py
EOF
md5sum -c md5.expected || { echo "CODE MISMATCH with job 6163 / 7193 code"; exit 2; }
R=results/neuron_seed_reproduction
SEEDS=${SEEDS:-"4738 4739"}
REPS=${REPS:-"1 2 3"}
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "restart count ${SLURM_RESTART_COUNT:-0}"
python -m pytest tests/test_neuron_graph.py tests/test_neuron_graph_slow_patch.py -x -q || exit 1

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

for REP in $REPS; do
  for S in $SEEDS; do
    B=$R/rep$REP/baseline/seed$S/full-synaptic-messages.pt
    step baseline-rep$REP-seed$S $B python scripts/train_neuron_graph_baseline_seed.py --seed $S --out-dir $R/rep$REP/baseline
    if [ -e $B ]; then
      step slow-patch-rep$REP-seed$S $R/rep$REP/slow-patch/seed$S/results.json python scripts/train_neuron_graph_slow_patch.py --seed $S --baseline-dir $R/rep$REP/baseline/seed$S --out-dir $R/rep$REP/slow-patch
    elif [ -z "$TERMED" ]; then
      echo "STEP slow-patch-rep$REP-seed$S exit code 3 (baseline missing)"; FAILED="$FAILED slow-patch-rep$REP-seed$S"
    fi
  done
done

if [ -n "$TERMED" ]; then
  echo "unfinished, requeue"; scontrol requeue $SLURM_JOB_ID; exit 0
fi
python scripts/score_neuron_seed_reproduction.py --root $R
SC=$?
echo "STEP score exit code $SC"
[ $SC -ne 0 ] && FAILED="$FAILED score"
if [ -n "$FAILED" ]; then echo "FAILED STEPS:$FAILED"; exit 1; fi
echo ALLDONE
