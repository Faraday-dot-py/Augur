#!/bin/bash
#SBATCH --job-name=augur-d5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=augur-d5-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=.
# JOB: bench | train (TAG, KEND, EXTRA) | eval (CANDS) | harness (HCK="tag:ckpt tag:ckpt")
case "${JOB:?}" in
bench)
  cd "$HOME/augur-contact"
  rm -f checkpoints_contact/D5bench.pt
  nvidia-smi --query-gpu=name,memory.total --format=csv
  for m in 16 8; do
    rm -f checkpoints_contact/D5bench.pt
    echo "=== bench micro $m"
    python scripts/scatter_contact_d5.py train --tag D5bench --k-end 50 --iters 6 --k-fixed 50 --warmup 1 --log-every 1 --save-every 1000 --micro $m || echo "BENCH FAILED micro $m"
  done
  ;;
train)
  cd "$HOME/augur-contact"
  python scripts/scatter_contact_d5.py train --tag "${TAG:?}" --k-end "${KEND:?}" ${EXTRA:-}
  ;;
eval)
  cd "$HOME/augur-contact"
  python scripts/scatter_contact_d5.py eval --cands "${CANDS:?}" ${EXTRA:-}
  ;;
harness)
  cd "$HOME/augur"
  mkdir -p checkpoints_d5 results/d5
  python scripts/scatter_regress.py run --ckpt checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt --out results/d5/regress_E.json || echo "HARNESS FAILED E"
  for hc in $HCK; do
    tag="${hc%%:*}"
    cp "$HOME/augur-contact/${hc#*:}" "checkpoints_d5/$tag.pt"
    python scripts/scatter_regress.py run --ckpt "checkpoints_d5/$tag.pt" --out "results/d5/regress_$tag.json" || echo "HARNESS FAILED $tag"
    python scripts/scatter_regress.py compare --baseline results/d5/regress_E.json --candidate "results/d5/regress_$tag.json" || true
  done
  ;;
esac
echo "=== ALL DONE"
