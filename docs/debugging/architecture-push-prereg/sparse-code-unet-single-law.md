# sparse-code-unet-single-law: pre-registration (stage 1)

Status: not run. Spec: docs/superpowers/specs/2026-11-09-architecture-push-twenty-experiments-design.md (experiment 18, section 10a overrides).

## Setup
- Candidate: single-law width-32 model with a top-k sparse code in the U-Net (SPARSE_K = 8 channels per cell kept; sites enc0-5, dec0-4; 11 sites).
- Controls: width 8 (`~/augur-single-law-network-sizing/ckpt/width8.pt`) and a width-32 dense reference passed by --ref32-ckpt. Available width-32 checkpoints on Polaris: job 7256 seed 4738 (default; had a loss spike), job 7442 seeds 4739 and 4741 (clean), 4740 (spiked), under `~/augur-single-law-network-sizing/ckpt_width32_spike/seed{4739,4740,4741}/width32.pt`. The choice is a needs-user decision; no seed filter applied.
- Candidate training seed 4738; same iteration count as the control (8000 iterations, 50 epochs).
- pp=2, knn=16, O(N). No hard-wired conservation. Sparsity is a learned code, not a physics prior.
- Eval seeds 9100/9200/9300; fresh 12000/12100/12200 before calling real; second training seed 4739 only after a pass.

## Lines (copied from spec)
- PASS: force error within 10% of the width-32 dense model AND top-5 ablation >= 5x the matched-random mean and above the random max on >= 2 of 3 held-out sets.
- KILL: force error > 1.30x or ablation < 2x.
- Example that fails PASS: ablation 3x.

## Operationalisation (assumptions, flagged)
- Force-error PASS: ratio <= 1.10 to the width-32 reference; KILL: ratio > 1.30.
- Ablation KILL: ratio < 2 on >= 2 of 3 sets (assumed reading of "ablation < 2x").
- Matched-random: random channel sets of equal size at the same sites. Ranking uses ablation effect over 11 sites x 32 channels (352 ablation evals).
- Ratio lines are quoted against the reference's own noise floor: if the reference is a spiked seed, its error is inflated and the 10% line is easier; this is why the seed choice is needs-user.
- Tie-break for ranking: equal effect resolved by lower site index then lower channel index.

## Plateau guard
INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations (epochs 46-50) is more than 3% below the mean over the preceding 10% window (epochs 41-45). One double-length extension allowed for candidate and control together.

## Prediction
Top-8 sparsity costs under 10% force error at width 32; top-5 ablation reaches 4-8x random. Stated before running; PASS is uncertain on the ablation line.

## Cost
Estimated 1.0-1.3 GPU-h (width-32 8000 iterations at about 0.308 s/it, sparse overhead, ranking, ablation). The spec says 0.7: flag to the supervisor.

## Needs-user
- Which width-32 checkpoint is the reference (REF32).
- Ablation KILL reading above.
