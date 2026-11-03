#!/bin/bash
#SBATCH --job-name=augur-tiered-drift-mechanism-diagnoses
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --nice=0
#SBATCH --output=/home/adamwebb/augur-tiered-drift-mechanism-diagnoses-%j.log
# Frozen e44b796 rollout dump + virial-and-loop-work-decomposition, restart-from-truth-drift-rate, integrator-compensation-test. No training; per-seed partial json is written after each seed (about 1 GPU-h total).
set -uo pipefail
export PYTHONUNBUFFERED=1
W=$HOME/augur_drift_diagnoses
mkdir -p $W && cd $W && tar xzf $HOME/drift_diag_code.tgz
export PYTHONPATH=$W
O=results/tiered_drift_mechanism_diagnoses
mkdir -p $O
python scripts/tier0_drift_diagnoses_tests.py --device cuda || { echo "TESTS FAILED"; exit 1; }
T0=$(date +%s)
python scripts/tier0_drift_diagnoses.py --new $HOME/augur_tiled_closed_loop/results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo.pt --out $O/diagnoses.json --dump-dir $O/dump
echo "SECONDS $(( $(date +%s) - T0 ))"
echo ALLDONE
