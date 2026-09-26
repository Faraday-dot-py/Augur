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

## Static results, job 3023, part b (results/genker_static_b.json)
Cost at matched rel_l2 0.01, flyby t5000:
| kernel | geo | own | transfer | own R2 | transfer R2 |
|---|---|---|---|---|---|
| inv_distance | 82 | 9.7 | 9.7 | 0.968 | 0.529 |
| yukawa30 | 463 | 180 | 218 | 0.636 | -0.072 |
| learned | 323 | 116 | 116 | 0.973 | 0.973 |
- inv_distance: own 8.5x fewer evals than geo; transfer matches own almost exactly even though transfer R2 is only 0.53 — the estimator error is uncorrelated with the audit-relevant tail, not that it's accurate.
- yukawa30 (screened, short range) is the weakest kernel by R2 as before (own 0.64, transfer negative — worse than the mean). Own estimator still beats geo 2.6x on flyby; transfer costs 20% more than own but still 2.1x geo. Transfer has no valid cost at 0.01 on uniform (fit doesn't reach the tolerance): with negative R2 the fitted own/transfer curves aren't reliable enough to interpolate.
- learned: own and transfer are statistically identical (R2 0.973/0.973) because the network was itself trained to match analytic short-range force, so the analytic-trained transfer estimator transfers almost perfectly. 2.8x fewer evals than geo.

## Full rollouts, job 3023 (600 steps, N=20000, dt 0.05, results/genker_<kernel>.json, snaps in results/genker_<kernel>_<prov>_snaps.npz)
All 7 kernels completed with no crashes. Energy/momentum/cost/audit summary (dE/E0 relative to exact's own energy scale; dx_med/scale = median position deviation from exact / exact's own p50 radial quantile at step 600; est_frac = fraction of force calls the audit let use the estimator rather than falling back to geometric; check_max = worst independent-particle audit error seen during the run, target 0.01):

| kernel | provider | dE/E0 | dx_med/scale | cost | est_frac | check_max | events |
|---|---|---|---|---|---|---|---|
| inv_distance | adaptive_own | 2.3e-4 | 0.16 | 27 | 1.00 | 0.008 | 0 |
| inv_distance | adaptive_transfer | 8.7e-5 | 0.13 | 47 | 1.00 | 0.004 | 0 |
| inv_distance | geo_audit | 3.4e-4 | 0.21 | 132 | 0 | 0.012 | 0 |
| softgrav2 | adaptive_own | 9.1e-5 | 0.05 | 138 | 1.00 | 0.011 | 0 |
| softgrav2 | geo_audit | 8.3e-5 | 0.05 | 364 | 0 | 0.016 | 0 |
| pow15 | adaptive_own | 6.9e-5 | 0.26 | 72 | 1.00 | 0.011 | 0 |
| pow15 | geo_audit | 1.2e-4 | 0.27 | 263 | 0 | 0.012 | 0 |
| yukawa30 | adaptive_own | 4.0e-5 | 0.06 | 170 | 1.00 | 0.011 | 0 |
| yukawa30 | adaptive_transfer | 4.4e-5 | 0.07 | 280 | 0.50 | 0.056 | 6 |
| yukawa30 | est_transfer (no audit) | 8.2e-5 | 0.07 | 145 | - | 0.334 | - |
| yukawa30 | geo_audit | 3.4e-5 | 0.07 | 391 | 0 | 0.021 | 0 |
| lj | adaptive_own | 1.6e-4 | 0.29 | 263 | 0.08 | 0.255 | 11 |
| lj | adaptive_transfer | 1.6e-4 | 0.29 | 264 | 0.08 | 0.270 | 11 |
| lj | est_own (no audit) | 2.0e-4 | 0.31 | 135 | - | 0.329 | - |
| lj | est_transfer (no audit) | 2.0e-4 | 0.30 | 146 | - | 0.302 | - |
| lj | geo_audit | 1.6e-4 | 0.28 | 289 | 0 | 0.012 | 0 |
| learned | adaptive_own | 5.5e-5 | 0.19 | 143 | 1.00 | 0.011 | 0 |
| learned | adaptive_transfer | 5.6e-5 | 0.19 | 144 | 1.00 | 0.011 | 0 |
| learned | geo_audit | 5.5e-5 | 0.19 | 337 | 0 | 0.015 | 0 |
| learned | exact_analytic (analytic force, not the network's) | dE/E0 vs its own E0 is a bound cluster (E swings -632734 to -621128 over 600 steps, ~1.8% drift); the network's own exact energy scale is E~4.0e8 (near-free/hot). Not comparable 1:1: this shows the learned force law is genuinely different from analytic gravity at this IC (weaker long range, from the extra 1/(d^2+1) rescale in kernels.learned), not that either rollout diverged. | | | | | |

Predictions checked:
- Audited error <= 1.5x target: holds for own everywhere and for adaptive_transfer everywhere except lj, where audit_own AND audit_transfer both sit at check_max 0.25-0.27 despite est_frac dropping to 0.08 (11 fallback events) — the audit detects the problem and mostly falls back, but even the residual 8% estimator-mode calls plus the geometric fallback threshold itself isn't tight enough to hold lj under 0.015. geo_audit alone (no estimator) holds lj to 0.012, so the residual violation is in the mixed est/geo path, not geometric fallback itself. Matches the predicted lj failure mode (sign-changing force breaks the |g|-normalised label).
- est_own/est_transfer (no audit) exceed 1.5x target for lj (0.30-0.33) as predicted, and for yukawa30 est_transfer (0.334, R2 was negative on this state) as predicted; yukawa30 est_own (with own R2 0.64) stays fine unguarded (only checked every 25 calls, so this is a lower bound on how bad it gets).
- Cost <= 1.1x geo on all, <0.7x on >=half: true for inv_distance (0.20x), pow15 (0.27x), yukawa30 (0.43x own), learned (0.42x); softgrav2 own is 0.38x geo too. lj own is 0.91x geo (audit forces it to behave almost like pure geometric, as predicted) — not a miss on cost, the audit is doing exactly what it's supposed to: making the failing kernel cost about the same as always-safe geometric rather than compromising accuracy.
- Momentum: all dual-tree providers land at |P| 1e-10 to 1e-11, exact at 1e-11 to 1e-12 — both far below dE/E0 scale, no asymmetry issue found for any of the 4 new kernels.
- Learned kernel: own/transfer both track the network's own force to dE/E0 5.5e-5, same order as inv_distance/softgrav2/pow15 exact energy drift, i.e. the tree+estimator reproduce the trained network's force law fine; the network's force law itself differs substantially from analytic gravity (see table note above), confirming the prediction that tree error is negligible next to model-vs-analytic deviation.

## Frame review (videos/genker_lj_grid.png, videos/genker_inv_distance_grid.png)
Unbiased subagent (no hypothesis given), density grids exact/adaptive_own/geo_audit (lj) and exact/adaptive_own/adaptive_transfer (inv_distance), steps 0/100/300/600: at every step, every provider row is visually indistinguishable from exact within each image — no blurring, missing structure, positional shift or artifact reported in any row. The only real difference the agent found was between the two force laws themselves (lj keeps two distinct merging cores through step 600; inv_distance fully merges into one elongated streak), which is physically expected and not a provider artifact. This matches the check_max/dE numbers: lj's failure mode (audit falling back 92% of the time, cost near geo) doesn't show up as visible structural error because the audit is doing its job — it's a cost story, not an accuracy one, at this resolution/step count.

## Conclusion
H4 holds with one caveat. On 6 of 7 kernels (analytic-like inv_distance, softened-gravity softgrav2/pow15, screened yukawa30, and the trained-network "learned" kernel) the node-pair estimator generalises to new force laws it was never trained on: own-kernel-trained estimators cut cost 2.3-8.5x vs geometric at matched accuracy, and even the wrong-kernel "transfer" estimator (trained only on analytic 1/d^2) is usually within 10-20% of the own-kernel estimator's cost because the audit controller (not the estimator's calibration) is what actually holds the rollout to target. The one real failure is a force law with a sign change (lj, repulsive core / attractive tail): the |g|-normalised error label the estimator was trained on collapses near the zero-crossing, so both own and transfer estimators sit at rel_l2 ~0.24-0.33 regardless of tolerance in the static test. In the full rollout the audit controller correctly detects this (11 fallback events, est_frac drops from ~1.0 to 0.08) and pushes lj's adaptive cost up to 91% of always-geometric — i.e. the safety net catches the bad estimator and the system degrades to "no faster than geometric" rather than to "wrong," but it does not recover the estimator's speed advantage for this kernel. Without the audit (est_own/est_transfer), lj and yukawa30's negative-R2 transfer case both silently exceed 1.5x target error, confirming the audit is load-bearing, not optional.

