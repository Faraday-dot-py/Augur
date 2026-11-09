#!/bin/bash
#SBATCH --job-name=augur-pair-term-adaptdeep-bruteforce-at-3e6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --dependency=afterany:7295
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-pair-term-adaptdeep-bruteforce-at-3e6-%j.log
# Locates the bodies where the default and adaptdeep pair-term outputs differ (max 1.37e-2) on the collapsed 3e6 Plummer cloud and brute-forces the top 20 in fp64. Seed 4738.
# Pass/kill: adaptdeep error on the argmax body <= 1.7e-6 PASS, >= 1e-4 KILL, between INCONCLUSIVE (applied by scripts/pair_term_adaptdeep_bruteforce_score.py).
# Estimate (untimed): the default call is one 222 s all-bodies-fallback call (job 7187), adaptdeep 0.4 s, cloud build and 20 brute-force bodies under a minute; about 6 min total.
# Needs in $HOME/augur_probe_deep: scripts/{scatter_field.py,pair_term_adaptive_depth_test.py,pair_term_adaptdeep_bruteforce_at_3e6.py,scatter_scaling_ops.py,pm_baseline_force_error.py} (uploaded copies, scatter_field.py as of commit 1810f32) and kern/E_frozen.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
D=$HOME/augur_probe_deep
ln -sfn $HOME/augur_nfr/kern $D/kern; cd $D || { echo "MISSING $D"; exit 2; }
export PYTHONPATH=.
for f in kern/E_frozen.pt scripts/scatter_field.py scripts/scatter_scaling_ops.py scripts/pm_baseline_force_error.py scripts/pair_term_adaptive_depth_test.py scripts/pair_term_adaptdeep_bruteforce_at_3e6.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
grep -q adaptdeep scripts/scatter_field.py || { echo "STALE scatter_field.py: no adaptdeep"; exit 2; }
[ "$(md5sum scripts/scatter_field.py | cut -d' ' -f1)" = 47b903e757cf1221dd71d10f47b7377a ] || { echo "scatter_field.py differs from commit 1810f32"; exit 2; }
[ "$(md5sum scripts/pair_term_adaptive_depth_test.py | cut -d' ' -f1)" = 1c8c97b74934c707431e03545271f015 ] || { echo "pair_term_adaptive_depth_test.py differs from commit 1810f32"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/pair_term_adaptdeep_bruteforce_at_3e6
mkdir -p $O
[ -s $O/DONE ] && { echo "skip: DONE exists"; echo ALLDONE; exit 0; }
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/pair_term_adaptdeep_bruteforce_at_3e6.py --ckpt kern/E_frozen.pt --device cuda --n 3000000 --core-scale 0.25 --out-dir $O >> $O/run.log 2>&1 &
PID=$!
wait $PID
rc=$?
[ $rc -eq 0 ] || { echo "exit $rc"; tail -20 $O/run.log; exit $rc; }
[ -s $O/DONE ] && [ -s $O/result.json ] || { echo "MISSING OUTPUT $O"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
