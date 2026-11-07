#!/bin/bash
#SBATCH --job-name=augur-scatter-core-drift-arm-a-fresh-held-out-seeds
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-scatter-core-drift-arm-a-fresh-held-out-seeds-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
DEADLINE=1794070800
REMAIN=$((DEADLINE-$(date +%s)))
[ $REMAIN -le 300 ] && { echo "DEADLINE 2026-11-07 09:00 PST: $REMAIN s left, not starting"; exit 3; }
past_deadline() { [ $((DEADLINE-$(date +%s))) -le 300 ]; }
dlrun() {
  local left=$((DEADLINE-$(date +%s)))
  [ $left -le 300 ] && { echo "DEADLINE: skipping $*"; return 124; }
  timeout -k 120 -s TERM $left "$@"
}
mkdir -p $HOME/augur_armA_fresh
cd $HOME/augur_armA_fresh
tar xzf $HOME/armA_fresh_code.tgz
export PYTHONPATH=.
GATE_CK=$HOME/augur_far/kern/E_frozen.pt
DRIFT_CK=$HOME/edge_ck/ghost_lsq.pt
for f in $GATE_CK $DRIFT_CK scripts/scatter_field.py scripts/scatter_conservation.py scripts/scatter_regress.py scripts/escape_roll.py scripts/armA_drift.py scripts/summarize_cons_compare.py scripts/scatter_core_drift_fresh_seeds.py scripts/scatter_core_drift_fresh_seeds_verdict.py tests/test_scatter_sym.py model/central_force.py results/far_kernels/r1b_mlp.pt; do test -f $f || { echo MISSING $f; exit 2; }; done
grep -q symc4 scripts/scatter_field.py || { echo MISSING symc4 in scripts/scatter_field.py; exit 2; }

FAIL=0
CHILD=
trap 'echo SIGTERM; [ -n "$CHILD" ] && kill -TERM $CHILD && wait $CHILD; exit 143' TERM
step() {
  name=$1; shift
  if past_deadline; then echo STEP $name skipped, deadline; DEADLINED=1; return; fi
  dlrun "$@" &
  CHILD=$!
  wait $CHILD
  rc=$?
  CHILD=
  echo STEP $name $rc
  if past_deadline && [ $rc -ne 0 ]; then DEADLINED=1; return; fi
  [ $rc -eq 0 ] || FAIL=1
}

R=results/armA_fresh
mkdir -p $R/cons_stock $R/cons_sym
step sym_test python -m pytest -q tests/test_scatter_sym.py
for V in stock sym; do
  OPT=default
  [ $V = sym ] && OPT=default,symc4
  [ -e $R/reg_$V.json ] || step regress_$V python scripts/scatter_core_drift_fresh_seeds.py regress run --ckpt $GATE_CK --opt $OPT --no-cons --out $R/reg_$V.json
  for REG in nr100 bh300; do
    [ -e $R/cons_$V/$REG.npz ] || step cons_${V}_$REG python scripts/scatter_core_drift_fresh_seeds.py conservation run --ckpt $GATE_CK --opt $OPT --kernel old --regimes $REG --steps 1000 --per-mult 32 --out-dir $R/cons_$V
  done
done
for S in 12100 12200 12300; do
  D=$R/n2e4_s$S
  mkdir -p $D
  step drift_roll_$S python scripts/escape_roll.py --ckpt $DRIFT_CK --n 20000 --seed $S --steps 1000 --arms S1,S1c4 --truth --resume --out $D
  step drift_measure_$S bash -c "python scripts/armA_drift.py $D S1 S1c4 > $D/drift.txt && cat $D/drift.txt"
done
step verdict python scripts/scatter_core_drift_fresh_seeds_verdict.py --cons-stock $R/cons_stock --cons-sym $R/cons_sym --reg-stock $R/reg_stock.json --reg-sym $R/reg_sym.json --drift-dirs $R/n2e4_s12100 $R/n2e4_s12200 $R/n2e4_s12300
echo DONE FAIL=$FAIL
[ -n "${DEADLINED:-}" ] && { echo "DEADLINE_STOP: unfinished steps skipped"; exit 4; }
exit $FAIL
