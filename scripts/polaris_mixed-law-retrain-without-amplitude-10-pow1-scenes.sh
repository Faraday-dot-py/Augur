#!/bin/bash
#SBATCH --job-name=augur-mixed-law-retrain-without-amplitude-10-pow1-scenes
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=student-pleiades
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-mixed-law-retrain-without-amplitude-10-pow1-scenes-%j.log
# Same recipe as near-field retrain job 3898 (scripts/polaris_nearfield/train.sh on branch near-field-retrain) plus --pow1-amp-max 3.1623 (pow1 amplitude 10^U(-1,0.5)).
# Needs $HOME/augur_nfr (scripts, model, kern/E_frozen.pt) as a base; local scripts/{scatter_multi,train_scatter_multi}.py are uploaded (scatter_field.py stays the augur_nfr copy used by job 3898, not the newer repo one) to $HOME/augur-mixed-law-upload and overlaid. Requeue-safe via --resume (ckpt every 500 its).
set -uo pipefail
export PYTHONUNBUFFERED=1
SEED=4738
TAG=mixed_pow1ampcap_s$SEED
W=$HOME/augur-mixed-law-pow1-amp-cap-baseline-scatter-field
U=$HOME/augur-mixed-law-upload
if [ ! -d $W/scripts ]; then
  mkdir -p $W
  cp -r $HOME/augur_nfr/scripts $HOME/augur_nfr/model $W/ || { echo "COPY FAILED"; exit 2; }
  mkdir -p $W/kern
fi
[ -f $W/kern/E_frozen.pt ] || cp -L $HOME/augur_nfr/kern/E_frozen.pt $W/kern/E_frozen.pt || { echo "MISSING kern/E_frozen.pt"; exit 2; }
for f in scatter_multi.py train_scatter_multi.py; do
  [ -e $U/$f ] || { echo "MISSING upload $f"; exit 2; }
  cp $U/$f $W/scripts/$f
done
cd $W || exit 2
export PYTHONPATH=$W
grep -q -- "--pow1-amp-max" scripts/train_scatter_multi.py || { echo "STALE train_scatter_multi.py"; exit 2; }
md5sum scripts/scatter_field.py scripts/scatter_multi.py scripts/train_scatter_multi.py kern/E_frozen.pt
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p ckpt
T0=$(date +%s)
PID=
trap 'kill -TERM $PID 2>/dev/null; wait $PID; exit 143' TERM
python scripts/train_scatter_multi.py --tag $TAG --seed $SEED --iters 80000 --pow1-amp-max 3.1623 --init-ckpt kern/E_frozen.pt --resume --ckpt-dir ckpt >> $TAG.log 2>&1 &
PID=$!
wait $PID
RC=$?
[ $RC -eq 0 ] || { echo "run exit $RC"; tail -20 $TAG.log; exit $RC; }
[ -s ckpt/$TAG.pt ] && [ -e ckpt/$TAG.done ] || { echo "MISSING OUTPUT ckpt/$TAG.pt or .done"; exit 1; }
echo "TOTAL_SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
