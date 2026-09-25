#!/bin/bash
#SBATCH --job-name=bounce-energy-ft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-energy-ft-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
mkdir -p results checkpoints

echo "=== baseline soup B probe2 ==="
python scripts/energy_probe2.py --device cuda --out results/energy_probe2_soupb.json | grep -v "^truth cons"

for cfg in "wide 1" "wide 8" "ctrl 8"; do
  set -- $cfg
  name=energy_ft_$1_k$2
  echo "=== finetune $name ==="
  python scripts/energy_finetune.py --mode $1 --unroll $2 --out checkpoints/$name.pt
  echo "=== probe2 $name ==="
  python scripts/energy_probe2.py --device cuda --checkpoint checkpoints/$name.pt --out results/energy_probe2_$name.json | grep -v "^truth cons\|^dataset"
  echo "=== rollouts $name ==="
  python scripts/energy_probe.py --device cuda --steps 200 --checkpoint checkpoints/$name.pt --only nb4_n20_g9.0,nb4_n317_g9.0,nb100_n100_g9.0,nb1000_n317_g9.0 --out results/energy_probe_$name.json | grep -v "^done nb"
done
echo ALLDONE
