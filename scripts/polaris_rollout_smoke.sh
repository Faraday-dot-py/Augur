#!/bin/bash
#SBATCH --job-name=augur-rollout-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=augur-rollout-smoke-%j.log

export PYTHONUNBUFFERED=1
cd "$HOME/augur"
export PYTHONPATH=.
mkdir -p results checkpoints
python scripts/est_train.py --kernel analytic --out checkpoints/est_analytic.pt
for ic in uniform flyby three plummer disk clumpy; do
  python scripts/nbody_rollout.py --ic $ic --n 5000 --steps 60 --force exact --diag-every 30 --snap-every 30 --tag smoke_${ic}_exact
done
python scripts/nbody_rollout.py --ic flyby --n 5000 --steps 60 --force adaptive --diag-every 30 --snap-every 30 --audit-every 5 --ref results/rollout_smoke_flyby_exact_snaps.npz --tag smoke_flyby_adaptive
python scripts/nbody_rollout.py --ic flyby --n 5000 --steps 60 --force geo_audit --diag-every 30 --snap-every 30 --audit-every 5 --ref results/rollout_smoke_flyby_exact_snaps.npz --tag smoke_flyby_geoaudit
python scripts/nbody_rollout.py --ic flyby --n 5000 --steps 60 --force bh --diag-every 30 --snap-every 30 --ref results/rollout_smoke_flyby_exact_snaps.npz --tag smoke_flyby_bh
python scripts/nbody_rollout.py --ic flyby --n 5000 --steps 60 --force mesh --diag-every 30 --snap-every 30 --grid 256 --ref results/rollout_smoke_flyby_exact_snaps.npz --tag smoke_flyby_mesh
python scripts/nbody_rollout.py --ic flyby --n 5000 --steps 60 --force exact --kernel yukawa --diag-every 30 --snap-every 30 --tag smoke_flyby_yukawa_exact
python scripts/est_train.py --kernel yukawa --out checkpoints/est_yukawa.pt
python scripts/nbody_rollout.py --ic flyby --n 5000 --steps 60 --force adaptive --kernel yukawa --diag-every 30 --snap-every 30 --audit-every 5 --ref results/rollout_smoke_flyby_yukawa_exact_snaps.npz --tag smoke_flyby_yukawa_adaptive
