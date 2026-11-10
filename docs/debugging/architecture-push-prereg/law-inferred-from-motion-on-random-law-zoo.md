# law-inferred-from-motion-on-random-law-zoo: pre-registration (stage 1)

Written before any run. Spec: experiment 12, section 10a overriding. Priority 2 (runs last of the law experiments).

Definitions. Law: tuple (a1, p1, a2, p2, inv_lam, eps, decaying) of scatter_far_general.LAWS; force magnitude (a1 r/(r^2+eps^2)^((p1+1)/2) + a2 r/(r^2+eps^2)^((p2+1)/2)) exp(-r inv_lam). Law zoo: hundreds of sampled smooth radial laws. Effective exponent: negative slope of log force against log r over r in 8-32 (2 for gravity). Code: the encoder's output that replaces theta. Tick-1 force error: force error at the first tick against the exact law.

## Stage 1 (what runs): data and code readiness only
The spec defines no cheap probe for stage 1, so none is run. Delivered: `scripts/law_zoo.py` (sampler, held-out exclusion, effective exponent, `LawEncoder`, linear exponent probe), 8 CPU tests, and a GPU data job that generates 300 zoo laws x 2 scenes plus 64 held-out-law scenes of exact truth (50 ticks) and checks counts reconcile.
Zoo sampler: p1 uniform 0.5-4.0; sign +1 (75%) or -1; Yukawa screening with probability 0.5 (inv_lam log-uniform 1/256 to 1/4); second opposite-sign term with probability 0.3; eps 0.5; amplitude scale log-uniform 0.1-10.
Held-out rule (builder-fixed): a law is excluded from training if its force-curve shape (log10 |F| at 16 radii 2-32, mean removed) is within rms 0.10 of pow1.5 or yuk32 (about p in [1.21, 1.79] and lambda in roughly [20, 50]). Held-out evaluation laws are drawn inside that region: 32 pow1.5-like (p in [1.4, 1.6]) and 32 yuk32-like (lambda in [28, 36]).
Encoder: DeepSets over bodies, 16 nearest neighbours, tokens (log r, radial component of the finite-difference acceleration, log(1+|a|)) over the first 4 observed ticks, 5-number code in the first theta slot, rest zero; output shape (B, 20) matches `MultiScatterField.set_law`. Dense `topk` neighbour search is used in stage 1 (O(N^2) per scene on 4 ticks); the full run must use the model's cell-list neighbour routine to keep O(N). Flagged, unverified.

## Control
Experiment 11's model (law-blind-unet-trunk), or main_s4739 (mixed-law, control M) if experiment 11 is KILL. Neither checkpoint path is verified in stage 1: the supervisor must name which exists after experiment 11's result; main_s4739 path is to be confirmed with polaris_ls before the full run (unverified). Single-law comparators: the existing per-law models used in scatter_multi_eval.py.

## Lines (verbatim, spec experiment 12)
- "PASS held-out pow1.5 tick-1 force at most 1.3x its single-law comparator AND held-out yuk32 at most 1.0x the control's own yuk32 error (the control already beats the yuk32 comparator) AND the code predicts the true exponent with R-squared at least 0.9 on held-out laws. KILL pow1.5 above 1.6x or yuk32 above 1.3x the control. Fails pass: pow1.5 1.45x."
- Operational R-squared (builder-fixed): linear map (with intercept) from code to effective exponent fitted on the training zoo laws, R-squared on the 64 held-out laws pooled (pow1.5-like and yuk32-like together).
- Stage 1 verdict: none. Stage 1 passes when the data summary reconciles (requested = generated = 664, accepted + rejected = 664, held-out distances below 0.10, training distances at or above 0.10, encoder code finite) and the 8 tests pass.

## Seeds and noise
Eval seeds 9100/9200/9300, fresh 12000/12100/12200 before any result is real; training seed 4738, 4739 only after a pass. Noise floor: the held-out yuk32 and pow1.5 errors of the control are the reference; seed noise 10-15% (so the 1.0x yuk32 line is inside the floor at one seed; pass provisional until 4739).

## Tie-break and selection
No selection among checkpoints (final weights). Zoo sampling is seeded by (seed, stream) so reruns reproduce; rejected draws are counted, not silently dropped. A ratio exactly on a line meets PASS and does not trigger KILL.

## Plateau guard (full run)
INCONCLUSIVE-undertrained if mean validation loss over the last 10% of iterations is more than 3% below the mean over the preceding 10% window; iteration count kept at the control's (to be read from the control's log before launch, unverified).

## Prediction (my estimate; the spec's own is about 20% pass chance)
Stage 1 data check passes. Full run: pow1.5 lands near 1.4-2x its comparator; the R-squared line is the likeliest failure (the held-out effective exponents span about 1.4-2.6). Probability of PASS: 15-20%.

## Constraints check
- pp=2, knn=16 in the model unchanged; the encoder is the only new part. No hard-wired conservation. Energy from the truth's E(t) in any energy metric. O(N): flagged above (encoder dense neighbour search in stage 1 only).
- Matched budget: full run keeps the control's iteration count; the encoder adds parameters, which is the change itself.

## Artifacts
Job script: scripts/polaris_law-inferred-from-motion-on-random-law-zoo-stage-1.sh. Results `results/law_zoo_stage1/data_summary.json` plus chunk files (about 0.15 GB). Estimated 0.2 GPU-h (untimed; spec stage-1 figure is 1). Full-run skeleton: none (no trainer yet).
