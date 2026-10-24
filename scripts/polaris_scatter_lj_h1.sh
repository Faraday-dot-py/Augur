#!/bin/bash
#SBATCH --job-name=augur-slj-h1
#SBATCH --qos=student-pleiades
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=augur-slj-h1-%j.log
# usage: sbatch --export=ARM=P scripts/polaris_scatter_lj_h1.sh
#   ARM = P (energy mesh only) | PR (energy mesh + soft-penalised residual) | SMOKE (tests + timing + truth gen)
# Needs ~/lj_h1_code.tgz (scripts/ tests/), truth npz in ~/augur_slj/results (seed 4738). Held-out seeds 4739, 4740 generated (3200 steps = 160 frames) on first run.
# Requeue-safe: train_scatter_lj.py --resume (checkpoint every 100 it + SIGTERM); eval only after training is done.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH=.
ARM=${ARM:-P}
ITERS=${ITERS:-6000}
D=$HOME/augur_h1
mkdir -p $D/results $D/ckpt
cd $D
tar xzf $HOME/lj_h1_code.tgz
for s in gas liquid solid coexist melt quench; do [ -f results/lj_$s.npz ] || cp $HOME/augur_slj/results/lj_$s.npz results/; done
[ -f ckpt/data_s4738_n320_f100_t2.5_h0.15.npz ] || cp $HOME/augur_slj/ckpt/data_s4738_n320_f100_t2.5_h0.15.npz ckpt/ 2>/dev/null
for seed in 4739 4740; do
  mkdir -p results/s$seed
  [ -f results/s$seed/lj_coexist.npz ] || python -m scripts.lj_sim --scenes gas,liquid,solid,coexist --steps 3200 --seed $seed --out results/s$seed/lj
done
PREF=results/lj,results/s4739/lj,results/s4740/lj
case $ARM in
  SMOKE)
    python -m pytest -q tests/test_scatter_lj.py tests/test_scatter_lj_energy.py
    python -m scripts.train_scatter_lj --tag smoke_P --ckpt-dir ckpt --energy --iters 60 --k-start 8 --k-end 32 --log-every 10 --val-every 1000 --ckpt-every 1000
    python -m scripts.train_scatter_lj --tag smoke_PR --ckpt-dir ckpt --energy --resid --iters 60 --k-start 8 --k-end 32 --log-every 10 --val-every 1000 --ckpt-every 1000
    python -m scripts.scatter_lj_drift --ckpt ckpt/smoke_PR_final.pt --tag smoke --prefixes results/lj --scenes liquid --frames 20
    echo SLJH1 SMOKE DONE
    exit 0;;
  ONLYEVAL) python -m scripts.scatter_lj_eval --ckpt ckpt/${TAG}_final.pt --tag h1_$TAG; echo SLJH1 EVALONLY DONE $TAG; exit 0;;
  P) FLAGS="--energy";;
  PR) FLAGS="--energy --resid";;
esac
python -m scripts.train_scatter_lj --tag $ARM --ckpt-dir ckpt $FLAGS --iters $ITERS --resume
rc=$?
[ $rc -eq 0 ] || { echo "train exit $rc"; exit $rc; }
python -m scripts.scatter_lj_drift --ckpt ckpt/${ARM}_final.pt --tag $ARM --prefixes $PREF
python -m scripts.scatter_lj_eval --ckpt ckpt/${ARM}_final.pt --tag h1_$ARM
echo SLJH1 DONE $ARM
