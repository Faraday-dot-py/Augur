#!/bin/bash
#SBATCH --job-name=augur-network-utilization-statistics
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --output=augur-network-utilization-statistics-%j.log
# Curiosity, inference only (no pass/kill). Rerun of 7242 after the topk SAE json lacked heldout_dead_fraction (now read with .get). Usage: sbatch --dependency=afterany:<slot 2 tail> this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur_nfr
export PYTHONPATH=.
for f in ckpt/main_s4738.pt ckpt/main_s4739.pt kern/E_frozen.pt scripts/network_utilization.py scripts/sae_interp.py; do test -f $f || { echo MISSING $f; exit 2; }; done
grep -q 'jr.get("heldout_dead_fraction")' scripts/network_utilization.py || { echo "STALE network_utilization.py"; exit 2; }
mkdir -p results/network_utilization
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/network_utilization.py --out results/network_utilization
rc=$?
echo network_utilization exit $rc
exit $rc
