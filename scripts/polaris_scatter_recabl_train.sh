#!/bin/bash
#SBATCH --job-name=augur-scatter-recabl-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-scatter-recabl-train-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
ARM="${ARM:?rec or norec}"
case "$ARM" in
  rec) VAR=ms_kp_pot_v_g128 ;;
  norec) VAR=ms_kp_pot_v_g128_norec ;;
  *) exit 1 ;;
esac
if [ ! -d "$HOME/augur-recabl" ]; then
  mkdir -p "$HOME/augur-recabl"
  (cd "$HOME/augur" && tar cf - --exclude=checkpoints --exclude=results --exclude='*.log' --exclude=.git --exclude=videos --exclude=web .) | tar xf - -C "$HOME/augur-recabl"
fi
cp "$HOME"/recabl_stage/*.py "$HOME/augur-recabl/scripts/"
cd "$HOME/augur-recabl"
export PYTHONPATH=.
CK=checkpoints/recabl_$ARM
RS=results/recabl_$ARM
mkdir -p $CK $RS
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
# stage 1 = the B-stage that produced init_B for Exp E (budgetB sfv: 900 s = 8087 it, steps 30, batch 16, k 4->20, lr 1e-3); fixed by iterations
python scripts/train_scatter_field.py --exp B --variant $VAR --train 2000 --steps 30 --batch 16 --k-start 4 --k-end 20 \
  --lr 1e-3 --iters 8087 --eval-steps 100 --log-every 100 --device cuda --seed 4738 --out-dir $RS/B --ckpt-dir $CK/B
# stage 2 = Exp E recipe exactly as polaris_scatter_bh_train.sh, with --iters 60769 in place of --time-budget 9000
python scripts/train_scatter_field.py --exp E --variant $VAR --train 1500 --steps 50 --batch 16 --k-start 4 --k-end 30 \
  --lr 5e-4 --iters 60769 --eval-steps 100 --log-every 100 --device cuda --seed 4738 \
  --init-ckpt $CK/B/B_$VAR.pt --out-dir $RS/E --ckpt-dir $CK/E
