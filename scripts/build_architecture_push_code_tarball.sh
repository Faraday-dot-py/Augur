#!/bin/bash
# Builds the code tarball for the shift-average/core-force diagnostics job (upload as shift_core_diag_code.tgz) and the staggered-double-grid full-run job (upload as staggered_double_grid_code.tgz).
# Uses tracked plus untracked-not-ignored files so it works before the commit; fails if a required file is missing.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-architecture_push_code.tgz}
git ls-files -co --exclude-standard scripts model augur.py conftest.py | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in scripts/shift_average_stage1.py scripts/core_force_component_verdict.py scripts/scatter_error_growth.py scripts/scatter_scenes.py scripts/scatter_field.py scripts/train_scatter_field.py \
  scripts/staggered_double_grid_eval.py scripts/staggered_double_grid_verdict.py scripts/plateau_guard.py model/__init__.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
