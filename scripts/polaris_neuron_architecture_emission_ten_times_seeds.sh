#!/bin/bash
#SBATCH --job-name=augur-neuron-arch-emission-10x
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
#SBATCH --output=augur-neuron-arch-emission-10x-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur-neuron"
tar xzf "$HOME/neuron_arch_code.tgz" -C "$HOME/augur-neuron"
cd "$HOME/augur-neuron"
export PYTHONPATH=.
for s in 4739 4740; do
  python scripts/train_neuron_graph_smoke.py --seed $s --stage1-iters 15000 --stage2-iters 15000 --out-dir results/neuron_graph_smoke_10x &
done
for v in absolute-reset-value-output in-degree-normalised-message-sum calibrated-emission-with-baseline-loss calibrated-emission-with-smoothed-van-rossum-loss; do
  python scripts/train_neuron_graph_loss_variants.py --variant $v --seed 4738
done
wait
echo ALLDONE
