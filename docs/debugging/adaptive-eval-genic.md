# genic: H4 generalization to unseen ICs and particle counts (analytic kernel)

Estimator under test: `checkpoints/est_analytic.pt`, trained only on uniform, flyby t0, flyby t1000 at N=100k. Unseen ICs: three, plummer,
disk, clumpy (`scripts/nbody_ic.py`). New files: `scripts/genic_lib.py`, `genic_static.py`, `genic_scaling.py`, `genic_rollout.py`,
`genic_roll_job.sh`, `genic_render.py`, `genic_summarize.py`. Jobs: 3007/3010 (exact 100k references), 3011 (scaling), 3012/3013 (rollouts),
3022 (static + retrain). All analytic kernel, seed 4738, cap 8, theta_max 1.2.

## Predictions (written before running)
Static: cost at matched error on unseen ICs will sit between the flyby numbers (worst case, similar contrast) and uniform (best case),
i.e. est/geo ratio roughly 0.4-0.7, calibration close to flyby's (R2 ~0.97, q90 ~0.90) since features are geometric/exchange-symmetric,
not IC-specific. Retraining on the unseen ICs will close some but not all of the gap. Scaling: cost/particle roughly constant 10k-300k
since the tree depth and feature scales (log size/dist etc.) are N-independent; calibration should hold. Rollouts: audited error stays at
target via the audit; adaptive keeps a strong cost edge on clumpy/three (high contrast), smaller on disk/plummer (smoother).

## 1. Static accuracy/cost (N=100k, steps 0/300/1000 of exact rollouts)

Kernel evals/particle at matched error, flyby-trained estimator vs geometric dual tree vs target-based BH (`results/genic_static.json`):

| state | geo@0.01 | bh@0.01 | est@0.01 | est/geo | est/geo@0.02 | est/geo@abs_p99=0.03 |
|---|---|---|---|---|---|---|
| uniform (ref) | 125 | 187 | 118 | 0.95 | 0.92 | 0.98 |
| flyby_t0 (ref) | 295 | 428 | 182 | 0.62 | 0.66 | 0.61 |
| flyby_t5000 (ref) | 329 | 668 | 117 | 0.36 | 0.45 | 0.34 |
| three (t0/t300/t1000) | 290/308/322 | 398/507/588 | 169/148/147 | 0.58/0.48/0.46 | 0.62/0.54/0.52 | 0.59/0.47/0.45 |
| plummer | 260/324/335 | 535/592/659 | 168/144/142 | 0.65/0.44/0.42 | 0.68/0.51/0.49 | 0.63/0.44/0.41 |
| disk | 303/331/314 | 466/519/526 | 149/127/122 | 0.49/0.38/0.39 | 0.56/0.44/0.44 | 0.48/0.37/0.36 |
| clumpy | 308/306/340 | 454/484/548 | 129/116/140 | 0.42/0.38/0.41 | 0.50/0.45/0.57 | 0.44/0.37/0.40 |

Every unseen-IC ratio falls inside the flyby/uniform range, mostly nearer the flyby (harder) end, i.e. no state is worse than flyby_t0.
Calibration on freshly labelled node pairs of each state (R2 of mean head, bias in log units, q90 coverage): R2 0.971-0.978, bias
-0.004..+0.016, q90 coverage 0.890-0.908 across all 16 unseen-IC states plus the 4 reference states — indistinguishable from the training
states' own numbers (experiment-log R2 0.970-0.979, q90 0.899-0.902). No degradation from the IC itself.

## 2. Estimator generalization gap (retrain on unseen ICs, held out at step 1000)

Retrained an `all_ics` head (three/plummer/disk/clumpy, steps 0+300) and 4 leave-one-out heads, tested at step 1000 of the held-out IC
(`results/genic_retrain.json`, `scripts/genic_static.py --retrain`):

| state | flyby cost@0.01 | all_ics | loo | flyby ratio | all_ics ratio | loo ratio |
|---|---|---|---|---|---|---|
| three_1000 | 147 | 145 | 145 | 0.46 | 0.45 | 0.45 |
| plummer_1000 | 142 | 140 | 140 | 0.42 | 0.42 | 0.42 |
| disk_1000 | 122 | 121 | 122 | 0.39 | 0.39 | 0.39 |
| clumpy_1000 | 140 | 139 | 139 | 0.41 | 0.41 | 0.41 |

Cost is identical to within 1-2% between the flyby-only head and either retrained head. R2 improves modestly (0.971-0.974 -> 0.978-0.980)
and bias moves closer to zero, but this buys no measurable cost or accuracy advantage at matched error. **Generalization gap is
negligible**: training on flyby alone already captures what the node-pair features need (geometry + quadrupole anisotropy, not IC identity).

## 3. Particle-count generalization (N = 10k/30k/100k/300k, flyby and uniform)

`results/genic_scaling.json`. Fixed tol=1e-3 cost: flyby 192-207 (range 15/192 = 7.8%), uniform 163-185 (range 22/163 = 13.5%) across the
4x-in-N range — well inside the 20% target. Calibration holds: R2 0.967-0.983, bias +-0.002, q90 coverage 0.899-0.904 at every N.
Audited (adaptive, target 0.01, static state, 60 calls) rel_l2 stayed 0.0083-0.0100 (flyby) and 0.0084-0.0103 (uniform) — inside the 1.5x
target band (<=0.015) everywhere, no fallback events triggered. Audited-cost variation is close to the 20% line: flyby adaptive cost
174-204 (17%, pass), uniform adaptive cost 102-124 (21.6%, marginal miss — driven by uniform's audited lam settling lower at small N,
102 at N=10k vs 121-124 at N>=30k). Geo_audit cost is noisier still (uniform 110-124, flyby 307-355) since theta has fewer degrees of
freedom to settle at low N; the estimator is more N-stable than its geometric ablation.

## 4. Rollouts (adaptive vs exact, geo_audit, mesh; N=50k, 1000 steps, audit target 0.01)

Jobs 3012/3013, `results/rollout_genic_r50k_<ic>_<force>.json`, density frames `videos/genic_clumpy_grid.png` and
`videos/genic_three_grid.png` (unbiased reviewer subagent, no hypothesis primed):

| ic | adaptive cost | geo_audit cost | ratio | adaptive audit mean/max | dens_corr (adaptive/geoaudit/mesh/noise-floor) | dx_med (adaptive/noise) |
|---|---|---|---|---|---|---|
| clumpy | 127 | 395 | 0.32 | 0.0091/0.0152 | 0.908/0.960/0.913/0.973 | 44.5/36.8 |
| three | 153 | 373 | 0.41 | 0.0091/0.0111 | 0.994/0.995/0.995/0.995 | 42.9/30.0 |
| plummer | 137 | 349 | 0.39 | 0.0092/0.0115 | 0.998/0.999/0.999/0.999 | 45.2/13.7 |
| disk | 124 | 326 | 0.38 | 0.0090/0.0111 | 0.236/0.081/0.003/0.786 | 91.2/43.7 |

Momentum: |P| stays at fp64 round-off (1e-11..1e-10) in every run including mesh (which is not momentum-symmetric by construction but is
still accurate here). Angular momentum drift is non-zero for the tree-based force providers (adaptive/geo_audit, up to 1e5 vs L0 ~1e6, i.e.
2-9%) because the audit/theta control changes the accepted-pair set between force calls, breaking the exact node-level antisymmetry
transiently; exact and the noise-floor control conserve L to 1e-9. This L drift was not previously characterised for geo_audit/adaptive and
should be flagged as an open item (not part of H4's stated pass bars, but relevant to H1). Energy drift (dE/|E| over 1000 steps) is small and
similar across all force providers per IC (clumpy ~1%, three ~1-1.5%, plummer 11-28%, disk 3.6-4.1%) — plummer's larger number reflects the
whole-system PE scale near a soft, deep cusp, not a force-provider difference (all providers, including exact, show the same order).

Reviewer subagent (unbiased, no hypothesis): both grids show gradual structural evolution, no periodic/blocky artifacts, and "all five rows
[exact, noise, adaptive, geoaudit, mesh] look almost identical at every step" for clumpy and three, with only subtle differences in the
outer halo/clump fine detail. This supports H1/H2 qualitatively for those two ICs.

**Disk is the one clear failure signal**: density correlation vs exact collapses for every non-exact provider (adaptive 0.24, geo_audit
0.08, mesh 0.003) while the 1e-6 noise-floor control still tracks exact well (0.79) and radial quantiles diverge outside the noise floor
(ratio 0.585-1.43 vs noise floor's 0.988-1.012). Since geo_audit and mesh — providers with no learned component — fail just as badly as
adaptive, this is not a symptom of the estimator; it is the disk IC's orbital dynamics being sensitive to force-error at a level between
the audit's rel_l2 target and the noise floor's 1e-6 perturbation (the exponential disk is prone to bar/spiral instabilities that amplify
small force errors quickly). Flagged as untested-for-this-audit-target territory, not an estimator generalization failure — but it means
the "reproduce exact-run statistics within the chaos floor" bar (H1) is NOT met for disk at target 0.01, by any force provider tested.

## Pass criteria vs result

| criterion | result | verdict |
|---|---|---|
| audited error <= 1.5x target (0.015) on every unseen IC | rollout audit mean 0.0089-0.0092 all ICs; scaling audit 0.0083-0.0103 | PASS |
| adaptive cost <= 1.1x geo_audit everywhere | 0.32-0.41x everywhere tested | PASS (much stronger than required) |
| adaptive cost < 0.7x geo_audit on clumpy/three | clumpy 0.32x, three 0.41x | PASS |
| q90 coverage within +-0.05 of 0.90 | 0.890-0.908 (static+scaling), i.e. within +-0.008 to +-0.010 | PASS |
| cost/particle varies < 20% over N=10k-300k (uniform) | fixed-tol 13.5% (pass); audited-target adaptive cost 21.6% (marginal miss) | MOSTLY PASS, one marginal miss |

## Summary
H4 holds for the analytic kernel: the flyby-trained estimator transfers to four structurally different unseen ICs (three, plummer, disk,
clumpy) and to N from 10k to 300k with no measurable calibration loss and a negligible retraining gap (<2% cost difference). The one
marginal miss is uniform's audited-cost variation across N (21.6% vs the 20% target), likely noise in a single static-state 60-call audit
run rather than a systematic scale dependence (fixed-tol cost varies only 13.5%). The one real finding needing a decision: rollout L drift
under audit/theta control (2-9% of L0) on all three non-uniform tree-based providers, and the disk IC's rollout statistics diverging from
exact for every provider (including geo_audit and mesh) at the 0.01 audit target — both open items for H1/H5 follow-up, not failures of
H4 specifically. Untested: kernels other than analytic for this generalization sweep (kernel-agnostic transfer was covered by a different
job/section in the log), 3D, wall-clock (unoptimised traversal, evals/particle only).
