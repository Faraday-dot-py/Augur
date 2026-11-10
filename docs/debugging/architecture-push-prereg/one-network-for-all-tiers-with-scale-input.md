# one-network-for-all-tiers-with-scale-input: stage 1 pre-registration

Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md (sections 5, 6, 7; 10a overrides). Registered before launch.

## Lines (copied from the spec)
Train on spans up to 256 units, evaluate at 1024 (two unseen tiers). Far-field relative error = sqrt(summed squared error / summed squared target) against the far field beyond radius 20.
- PASS: far-field relative error at 1024 at most 1.2x its own value at 256 AND its value at 256 at most 1.0x the per-tier control at 256.
- KILL: error at 1024 above 2.0x its own value at 256, OR its value at 256 above 1.15x the control.
- Fails the pass line: 1.5x (1024 over 256).

## Control
Per-tier-weights upper tiers (separate TierNet per tier, no scale input), trained in the same job with the same budget and seed, jointly. No pre-existing Polaris checkpoint is used: both arms start from scratch on the far target with a zero tier 0 (`--target far`, ZeroTier0).

## What is run
- Shared arm: one TierNet repeated at every tier, one extra constant input channel log2(tier cell width), and the output scaled by the cell width of the tier it is called at. Trained at span 256 (grid 512, 4 tiers, cell widths 2^k x 0.5) jointly (a shared net cannot be trained tier by tier).
- Control arm: per-tier weights, trained jointly with the same flags.
- Fixed numbers: width 32, 6 layers, G=16 tile, halo 4, 3000 iterations, batch 16, 160 scenes, 40 steps, lr 1e-3, far radius 20, seed 4738, validation every 100 iterations on 8 held-out scenes (seed 4738+5000). err, latent and learned-pool are off in both arms so all tiers have identical inputs.
- Eval (far mode): shared at span 256 (4 tiers) and span 1024 (6 tiers, scenes scaled with span), control at span 256, seeds 9100/9200/9300, 24 scenes, 100 steps, evaluation batch 2.

## Seeds and reality rule
Eval seeds 9100/9200/9300; fresh 12000/12100/12200 before a result is called real; training seed 4739 only after a pass.

## Tie-break rule
No argmin selection; the reported weights are the final iteration's. Equal values: lower seed index first.

## Ratio lines against the reference noise floor
Ratios are per-seed means of shared_1024 / shared_256 and shared_256 / control_256, printed with the denominator's seed spread (max - min) / mean. A ratio inside the spread is stated as such.

## Prediction (stated before launch)
Shared at 256 within about 1.0 to 1.2x the control (weight sharing costs a little). 1024 over 256 about 1.3 to 2x: two tiers are unseen and the scale channel takes values outside the training range. Most likely outcome: INCONCLUSIVE or KILL on the 1024 line; PASS is below even odds.

## Plateau guard (section 10a)
INCONCLUSIVE-undertrained if the mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window, applied to the shared arm; the iteration count stays the control's (3000). One extension at double length for both arms together. With 100-iteration validation spacing both windows hold about 3 points; the scorer fails loudly if a window is empty.

## Constraints check (spec section 5)
- pp=2, knn=16: not used in the far-field tiers (tier 0 is zero in this target). Not applicable.
- No hard-wired conservation: none. Satisfied.
- One change against the frozen control: shared weights plus scale input (one switch; the scale channel is what makes sharing possible). Satisfied.
- Matched budget: same iterations, batch, scenes, seed. Satisfied.
- No sweeps. Satisfied.
- Seeds: 4738 first. Satisfied.
- At most 2 GPUs: one GPU job.
- Over 1 h: full-state checkpoint every 250 iterations, --resume, SIGTERM saves. Satisfied.
- Diagnostic render plus fresh reviewer after the job: supervisor runs them.
- Requirements Til, O(N), Int: tiled hierarchy is linear in cells.
- Names: descriptive kebab-case.

## Cost
Spec 1 / 6 GPU-h. No logged rate for far-target joint training at span 256; the 6 GPU-h is the spec figure. Untested.

## Job
scripts/polaris_one-network-for-all-tiers-with-scale-input.sh. Full run draft: scripts/polaris_one-network-for-all-tiers-with-scale-input-full-run-draft.sh (not authorised).

## Caveats (from the fresh code review)
- The scale channel is a constant plane passed through the tile extraction; halo cells outside the box may read 0 rather than log2(cell width). Training and evaluation treat this identically, but the unseen tiers at span 1024 see a different edge value.
- The scorer applies the plateau guard to the shared arm's validation history only. If the guard fires, the double-length extension for both arms needs the control's history too; the scorer does not read it yet. Proposed fix at that point, not now.
