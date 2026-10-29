#!/bin/bash
#SBATCH --job-name=augur-neuron-graph-smoke
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-neuron-graph-smoke-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-neuron"
export PYTHONPATH=.
OUT=results/neuron_graph_smoke
python scripts/train_neuron_graph_smoke.py --seed 4738 --out-dir $OUT || exit 1
if grep -q '"verdict": "KILL"' $OUT/seed4738/results.json; then
  echo "seed 4738 verdict KILL, extra seeds skipped"
  exit 0
fi
python scripts/train_neuron_graph_smoke.py --seed 4739 --out-dir $OUT &
python scripts/train_neuron_graph_smoke.py --seed 4740 --out-dir $OUT &
wait
echo ALLDONE
