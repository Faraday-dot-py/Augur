#!/bin/bash
#SBATCH --job-name=augur-tier0-corrector
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-tier0-corrector-%j.log
# Oracle bound first (gate), then the corrector arms as concurrent processes on one GPU, then the held-out eval.
# Requeue-safe: finished outputs are skipped, training resumes from <tag>.pt (full state every 100 iterations and on SIGTERM).
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_corrector
mkdir -p $W && cd $W && tar xzf $HOME/corrector_code.tgz
export PYTHONPATH=$W
O=results/tier0_corrector
mkdir -p $O
S1=results/tier0_ext/tier0-matched-60k-iterations.pt
T0=$(date +%s)
[ -f $O/oracle_done ] || { python scripts/eval_tier0_corrector.py --mode oracle --s1 $S1 --out $O/oracle.json > $O/oracle.log 2>&1 && touch $O/oracle_done; }
echo "ORACLE_SECONDS $(( $(date +%s) - T0 ))"
python scripts/tier0_corrector_oracle_gate.py $O/oracle.json || { echo "ORACLE_KILL"; exit 0; }
declare -A EXTRA=( [corrector-position-loss]="" [corrector-position-plus-energy-similarity-loss]="--e-w 1.0" )
PIDS=()
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null; wait; exit 143' TERM
for tag in "${!EXTRA[@]}"; do
  [ -f $O/${tag}_best.pt ] && [ -f $O/${tag}.done ] && continue
  ( python scripts/train_tier0_corrector.py --s1 $S1 --tag $tag --out-dir $O --iters 2000 --resume ${EXTRA[$tag]} > $O/$tag.train.log 2>&1 && touch $O/$tag.done ) &
  PIDS+=($!)
done
wait
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
PIDS=()
python scripts/eval_tier0_corrector.py --mode eval --s1 $S1 --corr $O/corrector-position-loss_best.pt $O/corrector-position-plus-energy-similarity-loss_best.pt --out $O/eval.json > $O/eval.log 2>&1
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
