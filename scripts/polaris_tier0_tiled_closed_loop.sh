#!/bin/bash
#SBATCH --job-name=augur-tier0-tiled-closed-loop
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-tier0-tiled-closed-loop-%j.log
# Fine-tune the matched-60k tier 0 inside the tiled hierarchy (halo exchange of the recurrent field, closed-loop rollouts to 28 ticks), early-stop probes at
# iterations 1000 and 2500, then the held-out evals and the pre-registered scoring. Requeue-safe: training resumes from <tag>.pt (full state every 250
# iterations and on SIGTERM), finished outputs are skipped. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_tiled_closed_loop
mkdir -p $W && cd $W && tar xzf $HOME/tiled_closed_loop_code.tgz
export PYTHONPATH=$W
O=results/tier0_tiled_closed_loop
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur_tiered_c2_tier0_matched_60k.pt
TAG=tier0-tiled-closed-loop-halo
T0=$(date +%s)
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/$TAG.TRAIN_DONE ] && [ ! -f $O/$TAG.EARLY_KILL ]; then
  python scripts/train_tier0_tiled_closed_loop.py --init $INIT --data $HOME/t0n/data4 --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $O \
    --exchange halo --iters 5000 --resume >> $O/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "train exit $RC"; exit $RC; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
[ -f $O/$TAG.EARLY_KILL ] && { echo "EARLY_KILL: $(cat $O/$TAG.EARLY_KILL)"; exit 0; }
[ -f $O/$TAG.TRAIN_DONE ] || { echo "training incomplete"; exit 0; }
[ -f $O/${TAG}_eval.json ] || python scripts/eval_tier0_tiled_closed_loop.py --ckpt $O/$TAG.pt --exchange halo --out $O/${TAG}_eval.json > $O/$TAG.eval.log 2>&1
[ -f $O/${TAG}_hext.json ] || python scripts/hetero_ext_checks.py --ckpt $O/$TAG.pt --tag $TAG --out $O/${TAG}_hext.json > $O/$TAG.hext.log 2>&1
python scripts/score_tier0_tiled_closed_loop.py $O/${TAG}_eval.json $O/${TAG}_hext.json --out $O/${TAG}_score.json
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
