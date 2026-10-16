#!/bin/bash
#SBATCH --job-name=lrwarm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --dependency=afterany:3642
#SBATCH --output=lrwarm-%j.log
set -uo pipefail
export PYTHONUNBUFFERED=1
while [ "$(squeue -u $USER -h -t R -o %i | grep -v -x "$SLURM_JOB_ID" | wc -l)" -gt 0 ]; do echo "[$(date -Iseconds)] waiting for other running GPU jobs"; sleep 120; done
ROOT=$HOME/augur-lrw
mkdir -p $ROOT
cd $ROOT
tar xf $HOME/lrw_code.tar
mkdir -p results checkpoints
INITB=$HOME/augur/checkpoints/scatter_bh/init_B_ms_kp_pot_v_g128.pt
FROZEN=$HOME/augur/checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
export PYTHONPATH=.
ITERS=${ITERS:-60769}
V=ms_kp_pot_v_g128
arm() {
  a=$1; n=$2; extra=${3:-}
  python scripts/scatter_lrwarm.py --arm $a --iters $n --init-ckpt $INITB --out-dir results/lrw_$a$4 --ckpt-dir checkpoints/lrw_$a$4 $extra
}
post() {
  a=$1; ck=checkpoints/lrw_$a/E_$V.pt
  python scripts/scatter_rootcause_iter.py --variant $V --ckpts $ck,$ck.it10000,$ck.it30000 --out results/lrw_eval/iter_$a.json
  python scripts/scatter_rootcause_iter.py --variant $V --ckpts $ck --expb-seed 9500 --cluster-seed 9400 --out results/lrw_eval/iter_fresh_$a.json
  python scripts/scatter_regress.py run --ckpt $ck --out results/lrw_eval/regress_$a.json
  echo "=== compare $a"
  python scripts/scatter_regress.py compare --baseline results/scatter_baseline.json --candidate results/lrw_eval/regress_$a.json
  for r in 1 2 3; do python scripts/scatter_conservation.py run --ckpt $ck --opt default --steps 1000 --regimes nr100,bh300 --out-dir results/lrw_cons/${a}_r$r; done
}
mkdir -p results/lrw_eval
echo "=== smoke (30 it per arm, mech on ctl)"
for a in ctl w1 w2; do
  m=""; [ $a = ctl ] && m="--mech"
  arm $a 30 "$m" _smoke > results/lrw_smoke_$a.log 2>&1 || { echo "SMOKE FAIL $a"; tail -30 results/lrw_smoke_$a.log; exit 1; }
done
mkdir -p results/lrw_smoke; for a in ctl w1 w2; do cp results/lrw_${a}_smoke/lrwarm_${a}_s4738.json results/lrw_smoke/; done
python scripts/scatter_lrwarm_check.py results/lrw_smoke || { echo "KILL: warmup arm force error jumped >3x within 10 steps"; exit 2; }
grep -E "attribution|counterfactual" results/lrw_smoke_ctl.log
rm -rf checkpoints/lrw_*_smoke
echo "=== frozen E reference eval"
python scripts/scatter_rootcause_iter.py --variant $V --ckpts $FROZEN --expb-seed 9500 --cluster-seed 9400 --out results/lrw_eval/iter_fresh_frozenE.json
for a in ctl w1 w2; do
  m=""; [ $a = ctl ] && m="--mech"
  echo "=== train $a"
  arm $a $ITERS "$m" || exit 1
  echo "[$(date -Iseconds)] TRAINED $a"
done
for a in ctl w1 w2; do post $a; done
tar cf $HOME/lrw_results.tar results/lrw_ctl results/lrw_w1 results/lrw_w2 results/lrw_eval results/lrw_cons results/lrw_smoke results/lrw_smoke_ctl.log
echo "[$(date -Iseconds)] ALL DONE"
