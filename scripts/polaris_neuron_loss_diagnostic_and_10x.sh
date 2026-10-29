#!/bin/bash
#SBATCH --job-name=augur-neuron-loss-diag-10x
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
#SBATCH --output=augur-neuron-loss-diag-10x-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur-neuron"
tar xzf "$HOME/neuron_diag_code.tgz" -C "$HOME/augur-neuron"
cd "$HOME/augur-neuron"
export PYTHONPATH=.
python scripts/neuron_loss_diagnostic.py --seed 4738 &
python scripts/train_neuron_graph_smoke.py --seed 4738 --stage1-iters 15000 --stage2-iters 15000 --out-dir results/neuron_graph_smoke_10x &
wait
echo ALLDONE
