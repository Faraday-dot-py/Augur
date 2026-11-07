#!/bin/bash
# Builds the code tarball the tier-0 replay Polaris jobs extract (visited_replay_code.tgz; tiered_overnight_code.tgz for the third-seed, replay-on-second-base and sharpness-aware jobs).
# Includes results/loss_term_arms (frozen baselines the scorers read), results/tier0_tiled_closed_loop (frozen job 4060 eval) and results/tier0_second_training_seed (job 4818 eval, the replay reference).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT=${1:-visited_replay_code.tgz}
git ls-files scripts model augur.py conftest.py results/loss_term_arms results/tier0_tiled_closed_loop results/tier0_second_training_seed | tar czf "$OUT" -T -
LIST=$(tar tzf "$OUT")
for f in results/loss_term_arms/baseline_hier_eval_matched60k.json results/loss_term_arms/baseline_plain_eval_matched60k.json results/tier0_tiled_closed_loop/tier0-tiled-closed-loop-halo_eval.json \
  results/tier0_second_training_seed/tier0-tiled-closed-loop-halo-base-seed4739_eval.json model/__init__.py model/tiered/__init__.py \
  scripts/score_tier0_second_training_seed.py scripts/score_tier0_visited_state_replay.py scripts/score_tier0_fresh_seed_checkpoints.py scripts/score_tiered_sharpness_aware_minimisation_adam.py \
  scripts/eval_tier0_fresh_seed_checkpoints.py scripts/eval_tier0_tiled_closed_loop.py scripts/train_tier0_harvest.py scripts/train_tier0_tiled_closed_loop.py scripts/tier0_tiled_closed_loop_tests.py \
  scripts/hetero_ext_checks.py scripts/tier0_energy_metric_audit.py scripts/tier0_drift_diagnoses.py scripts/scatter_scenes.py; do
  grep -qx "$f" <<< "$LIST" || { echo "MISSING from tarball: $f"; exit 1; }
done
echo "$OUT $(wc -l <<< "$LIST") files $(du -h "$OUT" | cut -f1)"
