#!/bin/bash
# Same contents as scripts/build_replay_code_tarball.sh plus the uncommitted replay-mechanism scripts (git ls-files alone would drop them).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-replay_mechanism_code.tgz}
EXTRA="scripts/replay_mechanism_force_error_breakdown.py scripts/summarise_replay_mechanism_arms.py"
{ git ls-files scripts model augur.py conftest.py results/loss_term_arms results/tier0_tiled_closed_loop results/tier0_second_training_seed; for f in $EXTRA; do echo $f; done; } | sort -u | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739_eval.json \
  model/__init__.py scripts/train_tier0_tiled_closed_loop.py scripts/eval_tier0_tiled_closed_loop.py scripts/score_tier0_tiled_closed_loop.py scripts/train_tier0_harvest.py $EXTRA; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
