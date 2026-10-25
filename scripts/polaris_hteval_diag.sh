#!/bin/bash
#SBATCH --job-name=augur-hteval-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --output=augur-hteval-diag-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
W="$HOME/augur-hteval"
mkdir -p "$W/results/tiered"
cd "$W"
rm -rf scripts model tests
cp -r "$HOME/augur/scripts" "$HOME/augur/model" "$HOME/augur/tests" "$HOME/augur/conftest.py" .
cp -r "$HOME/augur-hteval-src/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/tests/." tests/
cp -r "$HOME/augur-hteval-src/extra/scripts/." scripts/
cp -r "$HOME/augur-hteval-src/extra/model/." model/
export PYTHONPATH=.
R=results/tiered
CK="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
T0="$HOME/augur-tier0/results/tier0_ext/armA.pt"
python -m pytest tests/test_tiered.py -q --tb=short 2>&1 | tail -15
python scripts/tiered_diag_blowup.py --model "t0ext:$T0" --out $R/diag_blowup_t0ext_armA.json || echo "FAILED diag"
python scripts/tiered_prof_sparse.py --model "scatter:$CK" --out $R/prof_sparse.json || echo "FAILED prof"
