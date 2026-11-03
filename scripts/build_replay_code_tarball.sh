#!/bin/bash
# Builds the code tarball the tier-0 replay Polaris jobs extract (visited_replay_code.tgz). Includes results/loss_term_arms (frozen baselines the scorer reads).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-visited_replay_code.tgz}
git ls-files scripts model augur.py conftest.py results/loss_term_arms | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json model/__init__.py scripts/score_tier0_second_training_seed.py scripts/train_tier0_harvest.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
