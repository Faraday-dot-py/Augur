#!/bin/bash
#SBATCH --job-name=augur-hteval-round2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --output=augur-hteval-round2-%j.log

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
S="$HOME/augur-tiered/checkpoints/tiered/tiered_S.pt"
J="$HOME/augur-tiered2/ckpt_J/tiered_J.pt"
B1="$HOME/augur-tier0/results/tier0_ext/armB1.pt"
A0="$HOME/augur-tier0/results/tier0_ext/armA0.pt"
python -m pytest tests/test_tiered.py -q --tb=short 2>&1 | tail -5
python scripts/tiered_holes.py exposure --arms "S=tiered:$S" "J=tiered:$J" --out $R/holes_exposure_J.json || echo "FAILED exposure J"
python scripts/tiered_holes.py respa --model "tiered:$J" --out $R/holes_respa_J.json || echo "FAILED respa J"
python scripts/tiered_arms.py run --arm C --model "tiered:$J" --out $R/arms_C_J.json || echo "FAILED arm C J"
for m in B1 A0; do
  eval ck=\$$m
  python scripts/tiered_eval.py --model "t0ext:$ck" --groups bias,seams,gravity,boundaries,exposure,generalisation --out $R/eval_t0ext_$m.json --grid-dir $R/grids_t0ext_$m || echo "FAILED eval $m"
  python scripts/tiered_diag_blowup.py --model "t0ext:$ck" --out $R/diag_blowup_$m.json || echo "FAILED diag $m"
done
python scripts/tiered_eval.py --model "tiered:$J" --groups gravity,exposure,generalisation,sparsity --out $R/eval_tiered_J.json --grid-dir $R/grids_tiered_J || echo "FAILED eval J"
python scripts/tiered_eval.py --model "tiered:$S" --groups generalisation --out $R/eval_tiered_S_gen.json || echo "FAILED eval S gen"
echo ROUND2DONE
