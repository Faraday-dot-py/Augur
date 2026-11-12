#!/bin/bash
# Builds the code tarball for the four stage-1 architecture-push jobs. Readability jobs extract it themselves; the bit-output and pair-context jobs expect it extracted into $HOME/augur_bit_output_64 and $HOME/augur_pair_context.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-augur-architecture-push-single-law-readability-code.tgz}
git ls-files scripts model augur.py conftest.py | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in model/__init__.py scripts/single_law_readability.py scripts/single_law_readability_probes.py scripts/single_law_network_sizing.py scripts/scatter_field.py scripts/scatter_multi.py scripts/scatter_far_general.py \
  scripts/bit_force_smoke.py scripts/bit_output_encoding_fixes.py scripts/bit_output_64_grid_long_range.py \
  scripts/train_scatter_ft.py scripts/pair_context_eval.py scripts/scatter_ft_data.py scripts/train_scatter_field.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
