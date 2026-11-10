#!/bin/bash
# Builds the code tarball for the architecture-push stage 1 jobs (ball-tree-particle-attention-tier-0, three-dimensional-sparse-voxel-scatter-field, finite-speed-local-field-medium). Run from the worktree after committing; not committed.
# Usage: scripts/build_architecture_push_stage1_code_tarball.sh <out.tgz>   (upload as $HOME/<experiment_name_with_underscores>_code.tgz, one copy per job)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-architecture_push_stage1_code.tgz}
git ls-files scripts model augur.py conftest.py | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in model/__init__.py model/ball_tree_attention.py model/sparse_voxel_scatter_field.py model/local_field_medium.py scripts/scatter_field.py scripts/train_scatter_field.py scripts/scatter_scaling.py scripts/scatter_scaling_ops.py \
  scripts/stage1_common.py scripts/stage1_preflight.py scripts/ball_tree_particle_attention.py scripts/three_dimensional_sparse_voxel_scatter_field.py scripts/finite_speed_local_field_medium.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
tar xzOf "$OUT" scripts/train_scatter_field.py | grep it_hook >/dev/null || { echo "tarball train_scatter_field.py has no it_hook"; exit 1; }
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
