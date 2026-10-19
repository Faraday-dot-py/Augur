#!/bin/bash
#SBATCH --job-name=augur-regce3
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd $HOME/augur-regce2
export PYTHONPATH=.
mkdir -p results/regce3
E=checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
[ -f $E ] || E=$HOME/E_ms_kp_pot_v_g128.pt
echo "=== unit tests"
python scripts/scatter_regress_contact_test.py results/regce2/h_E_baseline.json
echo "unit test exit $?"
echo "=== baseline E eager (--opt none) replicates 3"
python scripts/scatter_regress.py run --ckpt $E --out results/regce3/E_base_eager.json --exp E --opt none --replicates 3 --contact-energy --note "gate baseline, eager, median of 3"
echo "=== baseline E default opts replicates 3"
python scripts/scatter_regress.py run --ckpt $E --out results/regce3/E_base_default.json --exp E --replicates 3 --contact-energy --note "gate baseline, default opts, median of 3"
for B in eager default; do
  for M in C Bf1 Bf2 B2; do
    echo "=== compare $M vs E_base_$B"
    python scripts/scatter_regress.py compare --baseline results/regce3/E_base_$B.json --candidate results/regce2/$M.json --floors none
    echo "compare exit $?"
  done
done
echo "=== compare E(3645) vs E_base_default"
python scripts/scatter_regress.py compare --baseline results/regce3/E_base_default.json --candidate results/regce2/E.json --floors none
echo "=== done"
