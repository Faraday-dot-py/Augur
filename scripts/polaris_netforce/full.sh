#!/bin/bash
#SBATCH --job-name=augur-nf-armB
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=augur-nf-armB-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
mkdir -p "$HOME/augur_nf"
cd "$HOME/augur_nf"
tar xzf "$HOME/nf_code.tgz"
export PYTHONPATH=.
E=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
V=ms_kp_pot_v_g128
IT=${IT:-3000}
LAM=${LAM:-0.1}
nvidia-smi --query-gpu=name,memory.used --format=csv
mkdir -p results/netforce checkpoints/netforce
python scripts/train_scatter_netforce.py --arm B --lam $LAM --init-ckpt $E --iters $IT > results/netforce/trainB.log 2>&1 &
PB=$!
python scripts/train_scatter_netforce.py --arm C --lam 0 --init-ckpt $E --iters $IT > results/netforce/trainC.log 2>&1 &
PC=$!
trap 'kill -TERM $PB $PC 2>/dev/null; wait' TERM
wait $PB $PC
tail -n 3 results/netforce/trainB.log results/netforce/trainC.log
BK=checkpoints/netforce/B.pt; CK=checkpoints/netforce/C.pt
for S in 4738 9000 12000; do
  for A in E B C; do
    case $A in E) CKP=$E;; B) CKP=$BK;; C) CKP=$CK;; esac
    python scripts/escape_roll.py --ckpt $CKP --variant $V --n 20000 --seed $S --steps 400 --arms S1 --resume --probe-ticks 100,300 --out results/netforce/roll_${A}_$S
  done
  python scripts/core_drift_summary.py E=results/netforce/roll_E_$S,B=results/netforce/roll_B_$S,C=results/netforce/roll_C_$S S1 results/netforce/drift_$S.txt
done
python scripts/edge_eval.py --models E=$E,B=$BK,C=$CK --configs base=default --no-probe --out results/netforce/err_eval.json
python scripts/dE_seeds.py --ckpt-e $E --ckpt-b $BK --ckpt-c $CK --out results/netforce/dE --n-scenes 512 --arms E_nd1,E_nd2,B_nd1,C_nd1
python scripts/dE_seeds_summarize.py results/netforce/dE results/netforce/dE/summary.txt
echo NF DONE
