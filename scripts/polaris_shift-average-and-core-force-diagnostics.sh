#!/bin/bash
#SBATCH --job-name=augur-shift-average-and-core-force-diagnostics
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-shift-average-and-core-force-diagnostics-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
T0=$(date +%s)
E_CK=$HOME/E_ms_kp_pot_v_g128.pt
DRIFT_CK=$HOME/edge_ck/ghost_lsq.pt
OLD=$HOME/augur_armA_fresh/results/armA_fresh/n2e4_s12300

mkdir -p $HOME/augur_shift_core_diag/code_b $HOME/augur_shift_core_diag/code_c
cd $HOME/augur_shift_core_diag/code_b
tar xzf $HOME/shift_core_diag_code.tgz
cd $HOME/augur_shift_core_diag/code_c
tar xzf $HOME/armA_fresh_code.tgz
cp $HOME/augur_shift_core_diag/code_b/scripts/core_force_component_verdict.py scripts/
export PYTHONPATH=.

for f in $E_CK $DRIFT_CK results/far_kernels/r1b_mlp.pt scripts/escape_roll.py scripts/armA_drift.py scripts/core_force_component_verdict.py $OLD/truth.npz $OLD/truth.json; do test -f $f || { echo MISSING $f; exit 2; }; done
grep -q SYM_OPS scripts/scatter_field.py || { echo "MISSING SYM_OPS in code_c scripts/scatter_field.py"; exit 2; }
for f in scripts/shift_average_stage1.py scripts/scatter_error_growth.py scripts/scatter_scenes.py; do test -f $HOME/augur_shift_core_diag/code_b/$f || { echo MISSING code_b/$f; exit 2; }; done

FAIL=0
CHILD=
trap 'echo SIGTERM; [ -n "$CHILD" ] && kill -TERM $CHILD && wait $CHILD; exit 143' TERM
step() {
  name=$1; shift
  "$@" &
  CHILD=$!
  wait $CHILD
  rc=$?
  CHILD=
  echo STEP $name $rc
  [ $rc -eq 0 ] || FAIL=1
}

R=$HOME/augur_shift_core_diag/results
mkdir -p $R/shift_average_stage1 $R/core_force_components
cd $HOME/augur_shift_core_diag/code_b
if [ ! -e $R/shift_average_stage1/DONE ]; then
  step shift_average_stage1 python scripts/shift_average_stage1.py --ckpt $E_CK --out $R/shift_average_stage1/stage1.json
  [ $FAIL -eq 0 ] && touch $R/shift_average_stage1/DONE
fi

cd $HOME/augur_shift_core_diag/code_c
D=$R/core_force_components/n2e4_s12300
mkdir -p $D
cp -n $OLD/truth.npz $OLD/truth.json $D/
step core_roll python scripts/escape_roll.py --ckpt $DRIFT_CK --n 20000 --seed 12300 --steps 1000 --arms S1,S1c4 --probe-ticks 100,200,300,500,1000 --resume --out $D
step core_drift_reproduction bash -c "python scripts/armA_drift.py $D S1 S1c4 > $D/drift.txt && cat $D/drift.txt"
step core_verdict python scripts/core_force_component_verdict.py $D

for f in $R/shift_average_stage1/stage1.json $D/S1.json $D/S1c4.json $D/core_force_component_verdict.json; do [ -s $f ] || { echo OUTPUT MISSING $f; FAIL=1; }; done
echo TOTAL_SECONDS $(( $(date +%s) - T0 ))
echo DONE FAIL=$FAIL
exit $FAIL
