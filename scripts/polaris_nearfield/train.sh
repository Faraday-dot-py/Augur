#!/bin/bash
#SBATCH --job-name=augur-nfr-train
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=augur-nfr-train-%j.log
source $HOME/nfr_common.sh
SEED=${SEED:-4738}
trap 'kill -TERM $PID; wait $PID' TERM
python scripts/train_scatter_multi.py --tag main_s$SEED --seed $SEED --iters 80000 --resume --ckpt-dir ckpt > nfr_main_s$SEED.log 2>&1 &
PID=$!
wait $PID
echo "EXIT $?"
tail -n 3 nfr_main_s$SEED.log
