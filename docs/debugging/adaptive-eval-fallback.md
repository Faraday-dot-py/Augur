# Adaptive far field: H3 robustness of audit + geometric fallback (test id `fallback`, 2026-09-25)

## Predictions (written before any results)

1. yukawa30 with est_analytic: est (no audit) true force rel_l2 well above 3x target (0.03) from step 0 on flyby; adaptive falls back at call 0 (first audit, severe); geo error within 1.5x target afterwards; re-probe every 100 steps fails every time (about 9 failed probes per 1000 steps), and each failed probe costs about 10 steps (audit_every) of bad force. Uniform IC: unsure, may pass silently. yukawa30-trained estimator: no fallback.
2. Bias -2: controller compensates via lam (needs about 7.4x, lam_max 8, borderline); -4 and -6: fallback within 50 steps via severe/lam_max. Noise (sigma 1 in log units): heavy error tail, probably falls back or lives at high lam. Constant heads: fallback in the first few audits. Every fallback re-probes every 100 steps and the corrupted estimators fail the probe each time.
3. Healthy flyby/three/plummer/disk/clumpy at defaults (audit_k 1000, every 10, severe 3, fail_limit 4, lam_max 8, reprobe 100): under 1 false fallback per 1000 steps on most ICs; the lam controller equilibrium has about 19 percent of audits over target, so 4 consecutive fails (fail_limit) is the likely trigger.
4. Collapse to a dense core: est error rises; audit noticed within ~10 steps and fallback (or lam increase) follows; possibly the geo rule (theta 0.35 initial) also exceeds target in the dense core and theta adapts down.
5. Audit overhead at audit_k 1000, every 10: under 25 percent at N=20k; audit is exact_accel of 1000 targets (O(k N)), so relative cost falls with... it grows with N relative to O(N) force (k N vs N); at 100k possibly over.
6. |P|/sum|v| stays about 1e-13 or below everywhere with exchange-symmetric heads (noise head built symmetric); an asymmetric per-ordered-pair noise head would break momentum (extra check).
