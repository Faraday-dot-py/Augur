#!/bin/bash
#SBATCH --job-name=augur-hteval-holes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --output=augur-hteval-holes-%j.log

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
RR="$HOME/augur-tiered/checkpoints/tiered/tiered_R.pt"
T0="$HOME/augur-tier0/results/tier0_ext/armA.pt"
nvidia-smi --query-gpu=name,memory.used --format=csv
python -m pytest tests/test_tiered.py -q --tb=short 2>&1 | tail -30
python scripts/tiered_holes.py exposure --arms "S=tiered:$S" "R=tiered:$RR" --out $R/holes_exposure.json || echo "FAILED exposure"
python scripts/tiered_holes.py respa --model "tiered:$S" --out $R/holes_respa.json || echo "FAILED respa"
python scripts/tiered_holes.py sparse --model "tiered:$S" --out $R/holes_sparse.json || echo "FAILED sparse"
python scripts/tiered_holes.py recurrent --model "scatter:$CK" --out $R/holes_recurrent.json || echo "FAILED recurrent"
python scripts/tiered_holes.py calibration --model "tiered:$S" --out $R/holes_calibration.json || echo "FAILED calibration"
python scripts/tiered_arms.py run --arm C --model "tiered:$S" --out $R/arms_C.json || echo "FAILED arm C (S)"
python scripts/tiered_arms.py compare --dir $R || echo "FAILED compare"
python scripts/tiered_eval.py --model "t0ext:$T0" --groups bias,seams,gravity,boundaries,exposure,generalisation --out $R/eval_t0ext_armA.json --grid-dir $R/grids_t0ext || echo "FAILED t0ext eval"
python scripts/tiered_eval.py --model "tiered:$S" --groups gravity,exposure,generalisation,sparsity --out $R/eval_tiered_S.json --grid-dir $R/grids_tiered || echo "FAILED tiered eval"
