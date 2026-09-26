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

## Static results, job 3014 (100k flyby snapshots, 10k eval targets; results/genker_static_a.json)
Cost = kernel evals per target at matched rel_l2 (0.01), estimator alone, no audit.
| kernel | state | geo | own | transfer |
|---|---|---|---|---|
| softgrav2 | flyby t5000 | 328 | 103 | 112 |
| softgrav2 | uniform | 241 | 236 | 236 |
| pow15 | flyby t5000 | 147 | 38 | 39 |
| pow15 | uniform | 128 | 71 | 70 |
| lj | flyby t5000 | 274 | never | never |
| lj | uniform | 84 | 105 | 103 |
Held-out R2 (own / transfer, flyby t5000): softgrav2 0.97/0.94, pow15 0.975/0.87, lj 0.885/0.61.
- softgrav2 and pow15 (own cost 0.3x / 0.26x geo on flyby) confirm the prediction; transfer costs ~10% more for softgrav2 and equals own for pow15 (bias +0.39 in log error but q90 coverage 0.996: conservative).
- lj on flyby is a failure: own and transfer estimators sit at rel_l2 0.24-0.26 for every tolerance 1e-4..3e-2 (geo reaches 0.0044 at 633 evals). The error does not respond to the tolerance, so the estimator's predicted error for the offending pairs is far below the truth (sign-change labels, as predicted); an audit at 1e-2 will see rel_l2 0.24 and must fall back. On uniform lj is fine.
- Job 3014 also crashed on the learned-kernel timing run (kernels.learned f is 1D-only, pair_sums passes 2D); fixed with a reshaping wrapper in genker_kernels.make_kernel. lj exact: 0.045 s/step at N=20000.
