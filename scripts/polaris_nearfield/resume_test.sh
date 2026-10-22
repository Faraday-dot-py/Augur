#!/bin/bash
#SBATCH --job-name=augur-nfr-resume
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=augur-nfr-resume-%j.log
source $HOME/nfr_common.sh
rm -rf ckpt_rt
A="--tag rt --iters 400 --per-shape 300 --ckpt-every 50 --log-every 20 --ckpt-dir ckpt_rt --k-start 30 --k-end 30"
python scripts/train_scatter_multi.py $A &
PID=$!
sleep 75
kill -TERM $PID
wait $PID
echo "EXIT after SIGTERM: $?"
python scripts/train_scatter_multi.py $A --resume > rt2.log 2>&1 &
PID=$!
sleep 40
kill -TERM $PID
wait $PID
echo "EXIT 2: $?"
head -5 rt2.log
tail -5 rt2.log
nvidia-smi --query-gpu=memory.used --format=csv
echo RESUME TEST DONE
