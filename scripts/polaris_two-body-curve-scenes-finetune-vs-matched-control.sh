#!/bin/bash
#SBATCH --job-name=augur-two-body-curve-scenes-finetune-vs-matched-control
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --dependency=afterany:7299
#SBATCH --nice=100
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-two-body-curve-scenes-finetune-vs-matched-control-%j.log
# Fine-tune ckpt/main_s4739.pt 3000 its, lr 1e-4, k fixed 30, seed 4738: control (original data) then treatment (--two-body-frac 0.10), then held-out force-curve e_curve for pow1 pow3 lj, gravity err@20, and the pre-registered verdict.
# Needs $HOME/augur_nfr (scripts, model, ckpt/main_s4739.pt); local scripts/{scatter_multi,train_scatter_multi,scatter_multi_eval,two_body_curve_scenes_finetune_eval,rule_free_force_discovery,score_two_body_curve_scenes_finetune}.py are uploaded to $HOME/augur-two-body-upload and overlaid.
# scatter_field.py is deliberately NOT overlaid: the augur_nfr copy (43,222 bytes) is the one main_s4739 was trained with; the newer repo copy changes ScatterField constructor defaults.
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED=4738
ITERS=3000
W=$HOME/augur-two-body-finetune
U=$HOME/augur-two-body-upload
BASE=$HOME/augur_nfr/ckpt/main_s4739.pt
if [ ! -d $W/scripts ]; then
  mkdir -p $W
  cp -r $HOME/augur_nfr/scripts $HOME/augur_nfr/model $W/ || { echo "COPY FAILED"; exit 2; }
fi
[ -s $BASE ] || { echo "MISSING $BASE"; exit 2; }
for f in scatter_multi.py train_scatter_multi.py scatter_multi_eval.py two_body_curve_scenes_finetune_eval.py rule_free_force_discovery.py score_two_body_curve_scenes_finetune.py; do
  [ -e $U/$f ] || { echo "MISSING upload $f"; exit 2; }
  cp $U/$f $W/scripts/$f
done
cd $W || exit 2
export PYTHONPATH=$W
for f in scripts/rule_free_force_curves.py scripts/rule_free_force_discovery.py scripts/law_recovery_followup.py scripts/law_recovery_probe.py; do
  [ -e $f ] || { echo "MISSING $f"; exit 2; }
done
python scripts/two_body_curve_scenes_finetune_eval.py --help > /dev/null || { echo "EVAL IMPORT FAILED (scipy or rule_free_force_discovery missing)"; exit 2; }
python scripts/score_two_body_curve_scenes_finetune.py --help > /dev/null || { echo "SCORER IMPORT FAILED"; exit 2; }
grep -q -- "--two-body-frac" scripts/train_scatter_multi.py || { echo "STALE train_scatter_multi.py"; exit 2; }
grep -q -- "--init-multi" scripts/train_scatter_multi.py || { echo "STALE train_scatter_multi.py"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/train_scatter_multi.py scripts/scatter_multi_eval.py scripts/two_body_curve_scenes_finetune_eval.py scripts/rule_free_force_discovery.py scripts/score_two_body_curve_scenes_finetune.py $BASE
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p ckpt results/two_body_curve_scenes_finetune
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
run_arm() {
  TAG=$1
  shift
  python scripts/train_scatter_multi.py --tag $TAG --seed $SEED --iters $ITERS --lr 1e-4 --k-start 30 --k-end 30 --init-multi $BASE --resume --ckpt-dir ckpt "$@" >> $TAG.log 2>&1 &
  PID=$!
  wait $PID
  RC=$?
  [ $RC -eq 0 ] || { echo "$TAG exit $RC"; tail -20 $TAG.log; exit $RC; }
  [ -s ckpt/$TAG.pt ] && [ -e ckpt/$TAG.done ] || { echo "MISSING OUTPUT ckpt/$TAG.pt or .done"; exit 1; }
}
run_arm ft_control_s$SEED --two-body-frac 0
run_arm ft_twobody10_s$SEED --two-body-frac 0.10
python scripts/two_body_curve_scenes_finetune_eval.py --models base:$BASE,control:ckpt/ft_control_s$SEED.pt,treatment:ckpt/ft_twobody10_s$SEED.pt --out results/two_body_curve_scenes_finetune &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "eval exit $RC"; exit $RC; }
for n in base control treatment; do
  [ -s results/two_body_curve_scenes_finetune/$n.json ] || { echo "MISSING OUTPUT $n.json"; exit 1; }
done
python scripts/scatter_multi_eval.py --ckpt base=$BASE --ckpt control=ckpt/ft_control_s$SEED.pt --ckpt treatment=ckpt/ft_twobody10_s$SEED.pt --cases grav --out results/two_body_curve_scenes_finetune/gravity.json &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "gravity eval exit $RC"; exit $RC; }
[ -s results/two_body_curve_scenes_finetune/gravity.json ] || { echo "MISSING OUTPUT gravity.json"; exit 1; }
python scripts/score_two_body_curve_scenes_finetune.py --eval-dir results/two_body_curve_scenes_finetune --grav results/two_body_curve_scenes_finetune/gravity.json
RC=$?
[ $RC -eq 0 ] || { echo "score exit $RC"; exit $RC; }
[ -s results/two_body_curve_scenes_finetune/score.json ] || { echo "MISSING OUTPUT score.json"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
