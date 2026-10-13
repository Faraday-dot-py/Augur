#!/bin/bash
#SBATCH --job-name=augur-bfrozen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=augur-bfrozen-%j.log

set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/augur-bfrozen"
export PYTHONPATH=.
mkdir -p results/contact results/bfrozen checkpoints_contact refs
for f in "$HOME"/augur-contact/results/contact/evalset_{box15,box_var,post_circle,post_rect,disc,post_poly}_{9100,9200,9300}.pt "$HOME/augur-contact/results/contact/trainset_800_80.pt"; do
  [ -e "results/contact/$(basename "$f")" ] || cp "$f" results/contact/
done
ln -sfn "$HOME/augur-contact/checkpoints_contact/B2.pt" refs/B2.pt
ln -sfn "$HOME/augur-contact/checkpoints_contact/C.pt" refs/C.pt
ln -sfn "$HOME/augur-ftde/checkpoints_contact/Cfz.pt" refs/Cfz.pt
ln -sfn "$HOME/bounce-scenes/checkpoints" checkpoints
ls -la refs checkpoints/scatter_bh/E_ms_kp_pot_v_g128.pt
# STEPS: newline-separated argument lines (full command after python); a line starting with "!" aborts the job if it fails
while IFS= read -r step; do
  [ -z "$step" ] && continue
  echo "=== $step"
  if [ "${step:0:1}" = "!" ]; then
    python ${step:1} || { echo "GATE FAILED: $step"; exit 1; }
  else
    python $step || echo "STEP FAILED: $step"
  fi
done <<< "$STEPS"
