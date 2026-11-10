#!/bin/bash
#SBATCH --job-name=augur-sampled-force-distribution-control-stage-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --qos=student-pleiades
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-sampled-force-distribution-control-stage-1-%j.log
# Stage 1 of sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon: score the unchanged control (Exp E) on the tick-300 distribution metric against exact perturbed ensembles.
# Pre-registration: docs/debugging/architecture-push-prereg/sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon.md.
# Estimate about 0.5 GPU-h (untimed; 6 seeds x 3 perturbation sizes x 16 scenes x 17 float64 runs of 300 ticks). Requeue-safe: one json per (seed, perturbation size), skipped on rerun.
# Code tarball: bash scripts/build_architecture_push_stage1_code_tarball.sh, upload to $HOME. Supervisor adds --dependency at submission.
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_architecture_push_stage1_distribution
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
T=$HOME/architecture_push_stage1_code.tgz
[ -s $T ] || { echo "MISSING $T"; exit 2; }
[ -s $E ] || { echo "MISSING Exp E checkpoint $E"; exit 2; }
mkdir -p $W && cd $W && tar xzf $T || { echo "EXTRACT FAILED"; exit 2; }
export PYTHONPATH=$W
for f in scripts/score_scatter_distribution_control.py scripts/scatter_distribution_ensemble.py scripts/chaos_floor.py scripts/train_scatter_field.py scripts/scatter_field.py; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
md5sum scripts/scatter_distribution_ensemble.py scripts/score_scatter_distribution_control.py $E
nvidia-smi --query-gpu=name,memory.total --format=csv
O=results/scatter_distribution_control
mkdir -p $O
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/score_scatter_distribution_control.py --ckpt $E --out-dir $O >> $O/run.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "exit $RC"; tail -20 $O/run.log; exit $RC; }
[ -s $O/summary.json ] || { echo "MISSING OUTPUT $O/summary.json"; exit 1; }
n=$(ls $O/seed*_rel*.json | wc -l)
[ "$n" -eq 18 ] || { echo "EXPECTED 18 (seed, rel) results, found $n"; exit 1; }
grep -A6 '"summary"' $O/summary.json
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
