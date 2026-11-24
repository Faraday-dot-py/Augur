#!/bin/bash
# Builds the code tarball the three architecture-push stage-1 Polaris jobs extract (upload as $HOME/augur-architecture-push-stage-one-code.tgz). Not committed; commit the code first, git ls-files lists only committed files.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-augur-architecture-push-stage-one-code.tgz}
git ls-files scripts model augur.py conftest.py | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in model/__init__.py model/tiered/__init__.py model/tiered/tier.py model/tiered/hierarchy.py scripts/scatter_field.py scripts/scatter_multi.py scripts/scatter_multi_eval.py \
  scripts/train_scatter_multi.py scripts/scatter_far_general.py scripts/scatter_ladder.py scripts/scatter_ladder_stage1.py scripts/train_tiered.py scripts/gravity_sim.py \
  scripts/score_architecture_push_stage_one.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
