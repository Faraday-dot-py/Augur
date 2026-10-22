#!/bin/bash
#SBATCH --job-name=el-hfsmoke
#SBATCH --partition=gpu
#SBATCH --qos=student-pleiades
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=el-hfsmoke-%j.log
# Resume smoke: run polaris_el_heads_full.sh (uploaded as ~/el_hf_wrapper.sh) with 3000 it, ckpt every 20 it, SIGTERM it at it 100, rerun, expect "resumed from it".
export PYTHONUNBUFFERED=1
export NAME=hfsmoke SEED=4738 ITERS=3000 LOGEVERY=20 EVAL=0
rm -rf $HOME/augur-el-hfsmoke-4738
bash $HOME/el_hf_wrapper.sh > $HOME/hfsmoke_run1.log 2>&1 &
W=$!
until grep -q "^it 100 " $HOME/hfsmoke_run1.log 2>/dev/null; do sleep 1; kill -0 $W 2>/dev/null || break; done
kill -TERM $W
wait $W
echo "run1 exit $?"
grep -E "SIGTERM|^it |trainable|FAILED" $HOME/hfsmoke_run1.log | tail -4
bash $HOME/el_hf_wrapper.sh > $HOME/hfsmoke_run2.log 2>&1
grep -E "resumed|^trained|DONE|FAILED|trainable" $HOME/hfsmoke_run2.log
ls -la $HOME/augur-el-hfsmoke-4738/checkpoints/el_hfsmoke_s4738/
echo SMOKE DONE
