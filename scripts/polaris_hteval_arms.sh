#!/bin/bash
#SBATCH --job-name=augur-hteval-arms
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=03:00:00
#SBATCH --output=augur-hteval-arms-%j.log

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
CK="$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt"
R=results/tiered
nvidia-smi --query-gpu=name,memory.used --format=csv
python -m pytest tests/test_tiered.py -q 2>&1 | tail -30
python scripts/tiered_arms.py run --arm A --model "wide:$CK,grid=256,extent=128" --out $R/arms_A.json || echo "FAILED arm A"
python scripts/tiered_arms.py run --arm B --model "tiled:$CK,nt=4,tile=32,halo=16,far=1" --out $R/arms_B.json || echo "FAILED arm B"
if [ -n "${TIERED_NETS:-}" ]; then
  python scripts/tiered_arms.py run --arm C --model "tiered:$TIERED_NETS,$CK" --out $R/arms_C.json || echo "FAILED arm C"
fi
python scripts/tiered_arms.py compare --dir $R
python scripts/tiered_eval.py --model oracle --groups objects,gravity --seeds 4738 --out $R/eval_oracle_objects.json --grid-dir $R/grids || echo "FAILED oracle objects"
python scripts/tiered_eval.py --model constvel --groups objects --out $R/eval_constvel_objects.json --grid-dir $R/grids || echo "FAILED constvel objects"
python scripts/tiered_eval.py --model "scatter:$CK" --groups gravity,seams --seeds 4738 --out $R/eval_scatterE_overlay.json --grid-dir $R/grids_overlay || echo "FAILED overlay scatterE"
