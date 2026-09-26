# Adaptive far field: H3 robustness of audit + geometric fallback (test id `fallback`, 2026-09-25)

## Predictions (written before any results)

1. yukawa30 with est_analytic: est (no audit) true force rel_l2 well above 3x target (0.03) from step 0 on flyby; adaptive falls back at call 0 (first audit, severe); geo error within 1.5x target afterwards; re-probe every 100 steps fails every time (about 9 failed probes per 1000 steps), and each failed probe costs about 10 steps (audit_every) of bad force. Uniform IC: unsure, may pass silently. yukawa30-trained estimator: no fallback.
2. Bias -2: controller compensates via lam (needs about 7.4x, lam_max 8, borderline); -4 and -6: fallback within 50 steps via severe/lam_max. Noise (sigma 1 in log units): heavy error tail, probably falls back or lives at high lam. Constant heads: fallback in the first few audits. Every fallback re-probes every 100 steps and the corrupted estimators fail the probe each time.
3. Healthy flyby/three/plummer/disk/clumpy at defaults (audit_k 1000, every 10, severe 3, fail_limit 4, lam_max 8, reprobe 100): under 1 false fallback per 1000 steps on most ICs; the lam controller equilibrium has about 19 percent of audits over target, so 4 consecutive fails (fail_limit) is the likely trigger.
4. Collapse to a dense core: est error rises; audit noticed within ~10 steps and fallback (or lam increase) follows; possibly the geo rule (theta 0.35 initial) also exceeds target in the dense core and theta adapts down.
5. Audit overhead at audit_k 1000, every 10: under 25 percent at N=20k; audit is exact_accel of 1000 targets (O(k N)), so relative cost falls with... it grows with N relative to O(N) force (k N vs N); at 100k possibly over.
6. |P|/sum|v| stays about 1e-13 or below everywhere with exchange-symmetric heads (noise head built symmetric); an asymmetric per-ordered-pair noise head would break momentum (extra check).

## Setup actually run

N=100k (audit overhead also at 20k), 1000 steps (collapse 3000), defaults audit_k 1000 / every 10 / severe 3 / fail_limit 4 / lam_max 8 / reprobe 100, target 0.01. True force error is sampled every 5 steps (4000 independent exact targets; every 20 steps on the healthy-IC runs). Jobs 3020, 3027, 3036, 3039, 3042, 3045; results in `results/fallback_*.json`, summaries via `scripts/fallback_summarize.py` and `scripts/fallback_noise_summarize.py`. Harness bug found and fixed mid-run: the clumpy IC returns fewer than N bodies and the true-error sampler indexed N; fa_clumpy_adapt was rerun (job 3045) after the fix.

## Results (true force rel_l2 mean / max over sampled steps)

| Case | Outcome |
|---|---|
| Healthy est, 5 ICs (flyby, three, plummer, disk, clumpy) | 0 fallbacks, 0 probes. Error 0.0089-0.0091 mean, max 0.0106-0.0112. Interaction cost 117-144 vs geo_audit 329-420 (about 2.5-3x fewer). |
| Collapse IC, 3000 steps (adaptive, est, geo_audit) | No fallback. Error 0.0086-0.0092 mean, max 0.011, same as geo (0.0089). dE_max 0.13-0.17 in all three modes, so the dense core costs energy independent of the method. |
| Bias -2 head | Fallback at step 30 (lam 5.06 needed, 4 consecutive fails). 8 failed probes. |
| Bias -4, -6, const -10 | Fallback at step 0. 9 failed probes each, probe error 0.10 / 0.33 / 0.43. |
| Const -4, noise (sigma 1) head | No fallback. lam controller compensates: error 0.008-0.009, cost rises 136 to 181-188. |
| Asymmetric-noise head (200 steps) | No fallback, error 0.0093, but |P|/sum|v| = 7.6e-5 vs 1e-17 for every symmetric head. The audit does not see momentum loss. |
| Yukawa flyby | analytic-trained est: est-only mean 0.0176 max 0.315; adaptive falls back at step 280 (severe audit 0.091), then 5 failed probes. yukawa30-trained est: 0 fallbacks, 0.0092. |
| Yukawa uniform | analytic-trained est: est-only 0.092 (97 percent of steps over 3x target); adaptive falls back at step 0. yukawa30-trained est: 0.0067. |
| Momentum, all runs | symmetric heads 1e-17 to 1e-16, geo_audit same. |

Cost of a failed probe: each of the ~9 re-probes per 1000 steps runs the bad estimator for one audit interval (about 10 steps), so 9.5 percent of steps have true error over 3x target (bias -4/-6, const -10, uniform analytic). Mean error over the run is 0.018-0.05, max 0.10-0.49, all of it from probe steps.

Audit noise (full-N exact error vs 100 random k-subsets audits, 5 ICs, 255 samples): audit relative std 6.8 / 4.4 / 3.1 / 1.7 percent at k = 200 / 500 / 1000 / 3000. Misclassifying a step whose true error is within 0.7-1.4x target: 12.8 / 8.8 / 6.3 / 3.6 percent. Missing a truly over-target step: 29.7 / 20.1 / 14.7 / 7.9 percent. p_severe never above 0.

Audit overhead (fraction of a step at audit every 10): k=1000: 0.4-0.6 percent at 20k, 0.7-0.8 percent at 100k; k=3000: 1.2-1.6 percent at 20k, 2.0-2.5 percent at 100k. Force time per step, adaptive vs geo_audit: 0.044 vs 0.030 s at 20k, 0.107 vs 0.060 s at 100k (estimator inference costs more wall time than the saved interactions at these sizes; see the scaling test).

## Against predictions

1. Partly wrong. Yukawa analytic est on flyby did not fail at step 0 (fell back at 280; only the est-only run was clearly bad). Uniform did fail at step 0. Probe count (about 9/1000 steps) and cost per probe (~10 steps) correct.
2. Bias -2: fell back at step 30 (borderline predicted). -4/-6 and const -10: fell back at step 0 as predicted. Wrong: noise head and const -4 did not fall back; lam absorbs them.
3. Correct: 0 false fallbacks on all 5 ICs (fewer than 1 per 1000 steps).
4. Wrong: the collapse IC never stressed the estimator past target; no fallback, no theta adaptation needed.
5. Wrong by a large margin: 0.4-0.8 percent at k=1000, not 25 percent; even 100k stays under 3 percent at k=3000.
6. Correct: symmetric 1e-17; asymmetric noise 7.6e-5 (momentum leak with no audit signal).

## Recommendations

- Keep audit_k 1000 / every 10 (under 1 percent overhead). Raising to 3000 (2 percent) halves the near-threshold misclassification; worth it only if fail_limit decisions matter.
- Keep severe 3, fail_limit 4, lam_max 8: 0 false fallbacks on healthy ICs, real failures caught at step 0-280.
- Reduce the cost of a failed probe: audit the probe after 1-2 steps rather than a full interval, or run the probe as a shadow (estimator forces computed and audited, geo forces applied). This removes the ~9.5 percent of steps at 3-40x target error. Not implemented.
- reprobe_every 100 is fine for a failed head only if probes are cheap; with shadow probing, could shorten.
- Momentum is not protected by the audit: keep heads exchange-symmetric by construction (already true for the trained estimator); add a |P| check if an asymmetric head is ever used.
- The audit is a cross-check of force accuracy, not a substitute for a head validated on the target kernel: the analytic-trained head on Yukawa is caught (fallback) but a fallback run has geo-level cost (about 2.5-3x more interactions).
