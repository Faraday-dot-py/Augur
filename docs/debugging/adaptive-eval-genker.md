# adaptive-eval genker: H4 force-law generalisation in full rollouts (2026-09-25)

Files: scripts/genker_kernels.py (softgrav2 = d(d^2+4)^-1.5, pow15 = d(d^2+0.25)^-1.25 (f ~ d^-1.5 at range), lj = d s^-1.5 (1 - (4/s)^2), s = d^2+1: repulsive for d < 2, attractive tail),
genker_static.py, genker_rollout.py, polaris_genker_{a,b}.sh; results/genker_*.json.

## Predictions (written before any run)
- Audited force error (independent 2000-particle check) <= 1.5 x target 0.01 for every kernel and both estimators: yes for own; yes for transfer because the audit either
  lifts lam or falls back to geometric. Least sure: lj (f crosses zero: labels are normalised by |g| = |f(d)|, so exp(yhat) m|g| collapses near d = 2 and the estimator
  can under-predict absolute error there; audit sees it only if it moves rel_l2) and yukawa30 (own R2 was 0.5-0.65 previously).
- Transfer estimator wrong (yukawa30 and probably pow15/lj/inv_distance-optimistic): adaptive_transfer falls back within a few audits; est_transfer (no audit) runs silently with error > 1.5 x target for yukawa30.
- Own cost <= 1.1x geometric on all, < 0.7x on >= half (inv_distance ~0.1x, pow15/softgrav2/learned ~0.35x on flyby; yukawa30 and lj the likely misses).
- Momentum |P| ~1e-13 for all dual-tree providers (exact-pair symmetric); exact providers 1e-12 or better.
- Learned kernel: tree error << model-vs-analytic deviation (model was trained to reproduce analytic force at short-horizon).
