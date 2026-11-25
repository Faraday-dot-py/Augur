#!/bin/bash
#SBATCH --job-name=augur-tier0-tiled-energy-similarity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-tier0-tiled-energy-similarity-%j.log
# Fine-tune the tiled closed-loop tier 0 (e44b796 checkpoint) with an added energy-similarity term (model energy vs truth E(t) at the same tick, per scene
# and per cluster; never E(0)). Early-stop probes at iterations 750 and 1800 on dE@40 and the existing err/force/plain gates, then held-out evals and
# the pre-registered scorer. Requeue-safe (full state every 250 iterations and on SIGTERM). Usage: SEED=4738 sbatch --export=ALL,SEED=4738 this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED=${SEED:-4738}
W=$HOME/augur_tier0_tiled_energy_similarity_$SEED
mkdir -p $W && cd $W && tar xzf $HOME/tiled_energy_similarity_code.tgz
export PYTHONPATH=$W
O=results/tier0_tiled_energy_similarity
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt
TAG=tier0-tiled-energy-similarity-seed$SEED
T0=$(date +%s)
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/$TAG.TRAIN_DONE ] && [ ! -f $O/$TAG.EARLY_KILL ]; then
  python scripts/train_tier0_tiled_closed_loop.py --init $INIT --data $HOME/t0n/data4 --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $O \
    --exchange halo --iters 3000 --lr 1e-4 --k-start 16 --k-end 28 --w-energy 1.0 --energy-scale 0.02 --seed $SEED --val-seed ${VAL_SEED:-4739} --probe-at 750 1800 \
    --gate1-err 1.10 --gate1-force 1.10 --gate2-err 1.10 --gate2-bl 1.25 --gate1-dE 0.9 --gate2-dE 0.75 --resume >> $O/$TAG.train.log 2>&1 &
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
python scripts/tier0_energy_metric_audit.py --tier0 $E --m60k $HOME/augur_tiered_c2_tier0_matched_60k.pt --new $O/$TAG.pt --out $O/${TAG}_energy_audit.json > $O/$TAG.audit.log 2>&1
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
