#!/bin/bash
# Builds the code tarball the architecture-push stage 1 jobs extract (architecture_push_stage1_code.tgz); upload to $HOME. Tracked files only, so commit first.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-architecture_push_stage1_code.tgz}
git ls-files scripts model tests augur.py conftest.py | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in scripts/scatter_field.py scripts/scatter_ft_data.py scripts/scatter_mass_curriculum_data.py scripts/train_scatter_mass_curriculum.py scripts/train_scatter_field.py \
  scripts/train_scatter_timestep.py scripts/scatter_distribution_ensemble.py scripts/score_scatter_distribution_control.py scripts/law_zoo.py scripts/law_zoo_data_check.py \
  scripts/scatter_multi.py scripts/scatter_far_general.py conftest.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
