#!/bin/bash
# usage: genic_roll_job.sh ic [ic...]
cd "$HOME/augur"
export PYTHONPATH=. PYTHONUNBUFFERED=1
for ic in "$@"; do
  T=genic_r50k_$ic
  C="--ic $ic --n 50000 --steps 1000 --diag-every 100 --snap-every 50 --ckpt-every 500"
  python scripts/genic_rollout.py $C --force exact --tag ${T}_exact
  R="--ref results/rollout_${T}_exact_snaps.npz"
  python scripts/genic_rollout.py $C --force adaptive --tag ${T}_adaptive $R
  python scripts/genic_rollout.py $C --force geo_audit --tag ${T}_geoaudit $R
  python scripts/genic_rollout.py $C --force mesh --tag ${T}_mesh $R
  python scripts/genic_rollout.py $C --force exact --perturb 1e-6 --tag ${T}_noise $R
done
