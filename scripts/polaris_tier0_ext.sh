#!/bin/bash
#SBATCH --job-name=augur-tier0-ext
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=student-pleiades
#SBATCH --time=06:00:00
#SBATCH --output=polaris_mcp_jobs/slurm-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
ROOT="${ROOT:-$HOME/augur-tier0}"
mkdir -p "$ROOT" && cd "$ROOT" && [ -f "$HOME/tier0_code.tgz" ] && tar xzf "$HOME/tier0_code.tgz"
export PYTHONPATH=.
MODE="${MODE:-train}"
TAG="${TAG:-armA}"
T0="${T0:-ext=6,extapply,taper=8,bnd,halo=8}"
E_CKPT="${E_CKPT:-$HOME/E_ms_kp_pot_v_g128.pt}"
OUT="${OUT:-results/tier0_ext}"
mkdir -p "$OUT"
if [ "$MODE" = "est" ]; then
  python scripts/train_tier0_ext.py --t0 "$T0" --init-ckpt "$E_CKPT" --tag "$TAG" --out-dir "$OUT" --est
  exit $?
fi
python scripts/train_tier0_ext.py --t0 "$T0" --init-ckpt "$E_CKPT" --tag "$TAG" --out-dir "$OUT" --baseline "$E_CKPT" || exit $?
CK="$OUT/$TAG.pt"
for sc in sparse mid dense; do
  python scripts/hetero_tier_checks.py --ckpt "$CK" --t0 "$T0" --phase a --scenes $sc --out "$OUT/hetero_a_${TAG}_$sc.json"
done
for sc in sparse mid dense; do
  python scripts/summarize_hetero.py "$OUT" "${TAG}_$sc" 2>&1 | tail -60
done
