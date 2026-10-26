#!/bin/bash
#SBATCH --job-name=augur-showcase-solar_system_orbit_ft2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@600
#SBATCH --output=augur-showcase-solar_system_orbit_ft2-%j.log

# solar-system scene (star, 8 planets, moon, 200 belt particles), exact fp64 reference vs scatter-field orbit_ft2 over 2000 ticks; trajectories saved atomically at the end, a rerun skips if present; a few minutes on one GPU
# submit: sbatch --begin=2026-10-26T22:00:00 --nice=10000 scripts/polaris_showcase_solar_system_orbit_ft2.sh   (needs ~/showcase_code.tgz = tar of scripts/ model/)
if [ ! -d "$HOME/augur-showcase/scripts" ]; then mkdir -p "$HOME/augur-showcase" && tar -xzf "$HOME/showcase_code.tgz" -C "$HOME/augur-showcase"; fi
source "$HOME/augur-showcase/scripts/showcase_job_lib.sh"
showcase_setup
mkdir -p results/solar
showcase_run results/solar/showcase_solar_orbit_ft2_traj.npz python scripts/scatter_solar.py run --cands "orbit_ft2=E:ms_kp_pot_v_g128:$HOME/bounce-scenes/checkpoints_scenes/orbit_ft2.pt" --steps 2000 --hill-k 4.0 --tag showcase_solar_orbit_ft2 || exit 1
