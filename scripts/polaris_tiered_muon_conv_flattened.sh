#!/bin/bash
#SBATCH --job-name=augur-tiered-muon-conv-flattened
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --nice=1000
#SBATCH --signal=B:TERM@120
#SBATCH --output=/home/adamwebb/augur-tiered-muon-conv-flattened-%j.log
# Job 4060 recipe (matched-60k init, tiled closed-loop halo fine-tune, 5000 iterations, default probes and seeds) with only the optimizer changed:
# Adam -> Muon on 2-D weights and on 4-D conv kernels flattened to (out, in*kh*kw) (Adam on 1-D/biases, same lr and schedule). Requeue-safe: trainer resumes from <tag>.pt (full state every 250 iterations
# and on SIGTERM); finished outputs are skipped. Usage: sbatch this.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_tiered_muon_conv
mkdir -p $W && cd $W && tar xzf $HOME/muon_conv_code.tgz
export PYTHONPATH=$W
O=results/tiered_muon_conv_flattened
mkdir -p $O
E=$HOME/E_ms_kp_pot_v_g128.pt
INIT=$HOME/augur_tiered_c2_tier0_matched_60k.pt
D=$HOME/t0n/data4
TAG=tier0-tiled-closed-loop-halo-muon-conv-flattened-adam-rest
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json scripts/score_tiered_muon_conv_flattened.py model/__init__.py $E $INIT $D; do [ -e $f ] || { echo "MISSING $f"; exit 1; }; done
ls $D/hv3_s*.npz > /dev/null || { echo "MISSING harvest shards"; exit 1; }
python -c "import torch; assert hasattr(torch.optim, 'Muon'), torch.__version__; print('torch', torch.__version__)" || { echo "NO torch.optim.Muon"; exit 1; }
T0=$(date +%s)
[ -f $O/tests.done ] || { python scripts/tier0_tiled_closed_loop_tests.py --device cpu > $O/tests.log 2>&1 && touch $O/tests.done || { echo "TESTS FAILED"; tail -20 $O/tests.log; exit 1; }; }
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
if [ ! -f $O/$TAG.TRAIN_DONE ] && [ ! -f $O/$TAG.EARLY_KILL ]; then
  python scripts/train_tier0_tiled_closed_loop.py --init $INIT --data $D --pattern "hv3_s*.npz" --tier0 $E --tag $TAG --out-dir $O \
    --exchange halo --iters 5000 --optimizer muon-conv-flattened-adam-rest --resume >> $O/$TAG.train.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "train exit $RC"; exit $RC; }
fi
echo "TRAIN_SECONDS $(( $(date +%s) - T0 ))"
[ -f $O/$TAG.EARLY_KILL ] && { echo "EARLY_KILL: $(cat $O/$TAG.EARLY_KILL)"; exit 0; }
[ -f $O/$TAG.TRAIN_DONE ] || { echo "fine-tune incomplete (SIGTERM save)"; exit 143; }
set -e
trap 'echo "STEP FAILED at line $LINENO"' ERR
[ -f $O/${TAG}_eval.json ] || python scripts/eval_tier0_tiled_closed_loop.py --ckpt $O/$TAG.pt --exchange halo --out $O/${TAG}_eval.json > $O/$TAG.eval.log 2>&1
[ -f $O/${TAG}_hext.json ] || python scripts/hetero_ext_checks.py --ckpt $O/$TAG.pt --tag $TAG --out $O/${TAG}_hext.json > $O/$TAG.hext.log 2>&1
python scripts/score_tiered_muon_conv_flattened.py $O/${TAG}_eval.json --out $O/${TAG}_score.json
for f in $O/${TAG}_eval.json $O/${TAG}_hext.json $O/${TAG}_score.json; do [ -s $f ] || { echo "MISSING OUTPUT $f"; exit 1; }; done
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
