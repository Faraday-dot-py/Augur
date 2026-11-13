#!/bin/bash
#SBATCH --job-name=augur-summary-token-exchange-probe-with-early-stopping-and-planted-signal
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-summary-token-exchange-probe-with-early-stopping-and-planted-signal-%j.log
#SBATCH --dependency=afterany:9914
# summary-token-exchange-probe-with-early-stopping-and-planted-signal (inference-only measurement on the frozen replay checkpoint, job 4089; 1 GPU).
# Same probe as job 9914 plus early stopping on held-out train scenes, a ridge baseline, and a planted-signal validity gate (5% of error energy). Fresh eval seeds 12000/12100/12200, MLP train seeds 7100/7200/7300.
# Estimate about 0.2 GPU-h (job 9914 took 0:01:53 for 6 MLP fits; this fits 3 learners with up to 100 epochs; untimed assumption). Code tarball: bash scripts/build_tier0_stage1_probes_code_tarball.sh summary-token-exchange-probe-with-early-stopping-and-planted-signal-code.tgz, upload to $HOME.
# Requeue-safe: skips if the result json exists. The probe is not interruptible mid-run; a requeue restarts it (minutes).
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_summary_token_exchange_probe_with_early_stopping_and_planted_signal
mkdir -p $W && cd $W && tar xzf $HOME/summary-token-exchange-probe-with-early-stopping-and-planted-signal-code.tgz
export PYTHONPATH=$W
O=results/summary_token_exchange_probe_with_early_stopping_and_planted_signal
mkdir -p $O
CK=$HOME/augur_tier0_visited_state_replay_4738/results/tier0_visited_state_label_replay/tier0-visited-state-label-replay-seed4738.pt
for f in scripts/neighbour_tile_force_error_predictability_probe.py scripts/tier0_stage1_probe_common.py scripts/replay_mechanism_force_error_breakdown.py scripts/tier0_tiled_closed_loop.py \
  scripts/eval_tier0_tiled_closed_loop.py scripts/train_tiered.py scripts/exp12_hier.py scripts/scatter_field.py scripts/tier0_stage1_probes_tests.py model/tiered/__init__.py $CK; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$O/summary_token_exchange_probe_with_early_stopping_and_planted_signal.json
[ -s $OUT ] && { echo "already done: $OUT"; exit 0; }
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/neighbour_tile_force_error_predictability_probe.py --ckpt replay=$CK --seeds 12000 12100 12200 --early-stop --ridge --planted-share 0.05 --out $OUT &
PID=$!
wait $PID
rc=$?
[ $rc -eq 0 ] && [ -s $OUT ] || { echo "FAILED rc=$rc"; exit 1; }
echo "OK $OUT"
