#!/bin/bash
# Code tarball for the three stage-1 probes of the frozen tiled tier 0 (commit the probe scripts first; git ls-files then includes them). Never committed.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-tier0_stage1_probes_code.tgz}
git ls-files scripts model augur.py conftest.py | sort -u | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in model/__init__.py model/tiered/__init__.py scripts/scatter_field.py scripts/train_tiered.py scripts/exp12_hier.py scripts/tier0_tiled_closed_loop.py scripts/eval_tier0_tiled_closed_loop.py \
  scripts/replay_mechanism_force_error_breakdown.py scripts/tier0_stage1_probe_common.py scripts/dense_region_exact_force_ceiling_probe.py scripts/input_group_work_split_probe.py \
  scripts/neighbour_tile_force_error_predictability_probe.py scripts/tier0_stage1_probes_tests.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
