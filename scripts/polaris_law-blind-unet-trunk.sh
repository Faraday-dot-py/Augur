#!/bin/bash
#SBATCH --job-name=augur-law-blind-unet-trunk
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --dependency=afterany:SUPERVISOR_REPLACES_WITH_SLOT_TAIL_JOB_ID
#SBATCH --output=augur-law-blind-unet-trunk-%j.log
# Stage 1 (audit row 16): Exp E U-Net frozen, theta not given to the U-Net, kernel and pair networks trained, 80000 iterations (the control's count), seed 4738. Pre-registration: docs/debugging/architecture-push-prereg/law-blind-unet-trunk.md.
# Estimate 4.7-5.9 GPU-h (spec said 3.5): job 7296 ran 0.256 s/it at the end and 21214 s for 80000 its including data generation; the frozen trunk skips U-Net weight gradients (untested guess, 0.8x). No timing job was run.
# Requeue-safe: full-state checkpoint every 500 its with the validation history, SIGTERM saves and exits 143, --resume continues; sbatch again to resume.
# Needs $HOME/augur-architecture-push-stage-one-code.tgz, $HOME/augur_nfr/kern/E_frozen.pt, ckpt/main_s4739.pt, ckpt/cmp_pow1.5_single.pt, ckpt/cmp_yuk32_single.pt.
set -uo pipefail
export PYTHONUNBUFFERED=1
CODE=$HOME/augur-architecture-push-stage-one-code.tgz
W=$HOME/augur-law-blind-unet-trunk
N=$HOME/augur_nfr
E=$N/kern/E_frozen.pt
for f in $CODE $E $N/ckpt/main_s4739.pt $N/ckpt/cmp_pow1.5_single.pt $N/ckpt/cmp_yuk32_single.pt; do
  [ -s $f ] || { echo "MISSING $f"; exit 2; }
done
mkdir -p $W/ckpt $W/results || exit 2
SUM=$(md5sum $CODE | cut -d' ' -f1)
[ -e $W/.extracted-$SUM ] || { tar xzf $CODE -C $W && touch $W/.extracted-$SUM; } || { echo "EXTRACT FAILED"; exit 2; }
cd $W || exit 2
export PYTHONPATH=$W
grep -q -- "--freeze-trunk" scripts/train_scatter_multi.py || { echo "STALE train_scatter_multi.py: no --freeze-trunk"; exit 2; }
grep -q "law_blind" scripts/scatter_multi.py || { echo "STALE scatter_multi.py: no law_blind"; exit 2; }
grep -q "def ppe" scripts/scatter_field.py || { echo "STALE scatter_field.py: no ppe"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/train_scatter_multi.py $E
nvidia-smi --query-gpu=name,memory.total --format=csv
TAG=law-blind-unet-trunk-heads-only-s4738
ITERS=80000
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/train_scatter_multi.py --tag $TAG --seed 4738 --iters $ITERS --init-ckpt $E --law-blind --freeze-trunk --val-every 1000 --resume --ckpt-dir ckpt >> $TAG.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "train exit $RC"; tail -20 $TAG.log; exit $RC; }
[ -s ckpt/$TAG.pt ] && [ -e ckpt/$TAG.done ] || { echo "MISSING OUTPUT ckpt/$TAG.pt or .done"; exit 1; }
EV=results/law-blind-unet-trunk-stage-one-eval.json
if [ ! -s $EV ]; then
  python scripts/scatter_multi_eval.py --seeds 9100,9200,9300 --ckpt candidate=ckpt/$TAG.pt --ckpt main_s4739=$N/ckpt/main_s4739.pt --ckpt cmp_pow1.5_single=$N/ckpt/cmp_pow1.5_single.pt --ckpt cmp_yuk32_single=$N/ckpt/cmp_yuk32_single.pt --e-ckpt $E --cases grav,pow1,pow3,yuk8,yuk128,lj,HELD_pow1.5,HELD_yuk32 --out $EV >> $TAG.eval.log 2>&1 || { echo "eval failed"; tail -20 $TAG.eval.log; exit 1; }
fi
[ -s $EV ] || { echo "MISSING OUTPUT $EV"; exit 1; }
python scripts/score_architecture_push_stage_one.py law-blind --in $EV --val ckpt/$TAG.done --iters $ITERS --out results/law-blind-unet-trunk-stage-one-score.json || exit 1
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
