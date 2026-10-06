#!/bin/bash
#SBATCH --job-name=fallback-a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=fallback-a-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results
R="python scripts/fallback_run.py"
$R --name smoke_yuk --ic flyby --kernel yukawa --est checkpoints/est_yukawa30.pt --steps 40 --diag-every 20
$R --name smoke_noise --ic flyby --corrupt noise --steps 40 --diag-every 20 --noise-ks 200 1000
$R --name smoke_coll --ic collapse --steps 40 --diag-every 20
python scripts/fallback_overhead.py
for ic in flyby uniform; do
$R --name yuk_${ic}_estA --ic $ic --kernel yukawa --mode est
$R --name yuk_${ic}_adaptA --ic $ic --kernel yukawa
$R --name yuk_${ic}_adaptY --ic $ic --kernel yukawa --est checkpoints/est_yukawa30.pt
$R --name yuk_${ic}_geo --ic $ic --kernel yukawa --mode geo_audit
done
