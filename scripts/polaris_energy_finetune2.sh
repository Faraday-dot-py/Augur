#!/bin/bash
#SBATCH --job-name=augur-energy-ft2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=augur-energy-ft2-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
for cfg in "wall 1" "wall 8" "pair 1" "pair 8"; do
  set -- $cfg
  name=energy_ft_$1_k$2
  echo "=== finetune $name ==="
  python scripts/energy_finetune.py --mode $1 --unroll $2 --iters 3000 --batch 8 --scenes 600 --lr 5e-4 --out checkpoints/$name.pt
  echo "=== probe2 $name ==="
  python scripts/energy_probe2.py --device cuda --checkpoint checkpoints/$name.pt --out results/energy_probe2_$name.json | grep -v "^truth cons\|^dataset"
done
echo ALLDONE
