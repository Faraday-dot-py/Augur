#!/bin/bash
#SBATCH --job-name=euclid-fs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=euclid-fs-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
ROOT=$HOME/augur-ef
mkdir -p $ROOT
cd $ROOT
if [ ! -f scripts/train_scatter_field.py ]; then tar xf $HOME/ef_code.tar; fi
mkdir -p results checkpoints/ef
[ -f results/scatter_baseline.json ] || cp $HOME/augur/results/scatter_baseline.json results/
INITB=$HOME/augur/checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt
FROZEN=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
export PYTHONPATH=.
ITERS=${ITERS:-60769}
train() {
  arm=$1; iters=$2; tag=$3
  python scripts/train_scatter_field.py --exp E --variant ms_kp_pot_v_g128 --mode train --init-ckpt $INITB --train 1500 --steps 50 --iters $iters --batch 16 --k-start 4 --k-end 30 --lr 5e-4 --device cuda --seed 4738 --rloss $arm --out-dir results/ef_$tag --ckpt-dir checkpoints/ef_$tag || return 1
}
post() {
  tag=$1; ck=checkpoints/ef_$tag/E_ms_kp_pot_v_g128.pt
  python scripts/scatter_losstest.py --setup warm --arm sq --tag _$tag --eval-ckpt $ck --out results/ef_eval --ckpt checkpoints/ef_eval --iters 0
  python scripts/scatter_regress.py run --ckpt $ck --out results/ef_eval/regress_$tag.json
  echo "=== compare $tag"
  python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate results/ef_eval/regress_$tag.json
  python scripts/scatter_conservation.py run --ckpt $ck --opt default --steps 1000 --regimes nr100,bh300 --out-dir results/ef_cons/$tag
}
echo "=== smoke (30 it per arm)"
for arm in sq euclid; do
  train $arm 30 smoke_$arm > /dev/null 2>&1 || { echo "SMOKE FAIL $arm"; train $arm 30 smoke_$arm 2>&1 | tail -20; exit 1; }
done
rm -rf checkpoints/ef_smoke_* results/ef_smoke_*
echo "=== frozen E reference eval"
python scripts/scatter_losstest.py --setup warm --arm sq --tag _frozenE --eval-ckpt $FROZEN --out results/ef_eval --ckpt checkpoints/ef_eval --iters 0
echo "=== train control (sq)"
train sq $ITERS sq || exit 1
echo "[$(date -Iseconds)] CONTROL TRAINED"
echo "=== train euclid"
train euclid $ITERS euclid || exit 1
echo "[$(date -Iseconds)] EUCLID TRAINED"
post sq
post euclid
tar cf $HOME/ef_results.tar results/ef_sq results/ef_euclid results/ef_eval results/ef_cons
echo "[$(date -Iseconds)] PRIMARY DONE"
