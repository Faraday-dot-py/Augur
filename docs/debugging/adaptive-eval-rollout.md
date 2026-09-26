# Adaptive far field: long-rollout evaluation (test id `rollout`, 2026-09-25)

Tests H1 (accuracy) and H2 (efficiency) of ALF (adaptive learned far field) against exact, geo_audit, est, geo, bh, mesh in N=100k 2D rollouts (analytic softened gravity, dt 0.05, KDK leapfrog, unit masses).

## Setup
- Runner: `scripts/rollout_nbody.py` (copy of `nbody_rollout.py` adding `--seed`, `--perturb`, `--max-wall`, full adaptive-state checkpoint/resume, `vsum` diagnostic). Queue: `scripts/rollout_queue.sh` + `scripts/rollout_tasks.txt` (resumable, one <45 min Slurm job at a time). Analysis: `scripts/rollout_analyze.py`; frames: `scripts/rollout_render.py`.
- Scenarios: flyby seed 4738 x 4000 steps (exact, adaptive, geo_audit, est, geo, bh, mesh1024, mesh2048), noise-floor control (exact + 1e-6 uniform position perturbation, 500 steps), flyby seed 12000 (exact, adaptive, geo_audit, mesh1024), uniform collapse 3000 steps (exact, adaptive, geo_audit, mesh1024).
- Audit target 0.01, audit every 10 steps on 1000 random particles, default cap 8, tol 1e-3, estimator `checkpoints/est_analytic.pt`.
- Smoke (job 3008, N=5000, 60 steps): resume path, perturbation, seed, mesh all run; adaptive at N=5000 showed larger energy drift than exact (-84335 vs -84461 at step 40 from -84490), noted as an early flag for the 100k run.

## Predictions (written before the runs)
1. adaptive |dE| <= geo_audit |dE| and <= mesh |dE|: likely true vs mesh; roughly equal to geo_audit (both audited to the same 0.01 error; energy drift is set by force-error bias, not the learned part). Real risk: both exceed exact's own integrator drift by a visible margin (N=5000 smoke already shows this).
2. |P|/sum|v| <= 1e-10 for adaptive/geo_audit/est/geo (momentum-symmetric dual tree, fp64: expect 1e-12..1e-13); bh and mesh not conserving (~1e-4..1e-3).
3. Density corr and radial quantile ratios of adaptive/geo_audit within the noise-floor spread: uncertain. The floor is chaos-amplified 1e-6, whereas 1% force error is a much larger perturbation, so by step ~500+ pointwise-derived statistics may fall outside the floor spread while still statistically equivalent; I expect density corr to sit below the floor after the close encounter.
4. adaptive cost/particle <= geo_audit on flyby (prior tests: 0.27-0.34x), ~1.0-1.25x on uniform collapse (may exceed the 1.1x limit at early uniform states).
5. No fallback events on flyby seed 4738; possible late in the run as halo extent leaves the training distribution (est trained on t0/t1000 flyby + uniform only). Collapse: possible.
6. Mesh1024 conserves poorly on the collapse IC (density contrast).

## Results
All 16 tasks completed (Polaris jobs 3021, 3026, 3038, 3041, 3044, 3047; roughly 3.5 GPU-hours incl. resumes). Scripts: `rollout_analyze.py`, `rollout_tabulate.py`, `rollout_mincorr.py`, `rollout_render.py`. Figures: `videos/rollout_figs/`, `videos/rollout_flyby_grid.png`, `videos/rollout_uniform_grid.png`.

Caveat on the reference: exact direct-sum at dt 0.05 itself drifts 11.4% of |E0| (flyby, 4000 steps), 12.1% (seed 12000), 2.0% (uniform). "dE vs exact" below measures agreement with that reference, not physical energy conservation.

### Per-step cost (N=100k, one GPU)
| provider | flyby s/step | interactions/particle | uniform s/step |
|---|---|---|---|
| exact | 0.562 | all pairs | 0.562 |
| adaptive | 0.096 | 138 | 0.092 (147) |
| geo_audit | 0.104 | 342 | 0.083 (263) |
| est | 0.093 | 131 | |
| bh | 0.269 | | |
| mesh1024 | 0.012 | | 0.014 |
| mesh2048 | 0.016 | | |

### Conservation (max over run)
| provider | flyby dE vs exact (frac of E0) | s12000 | uniform | |P|/sum|v| flyby | L drift flyby | L drift uniform |
|---|---|---|---|---|---|---|
| adaptive | +0.0218 | +0.0165 | -0.0117 | 6e-17 | 3e-4 | 1.5e-2 |
| geo_audit | +0.0194 | +0.0154 | -0.0112 | 4e-17 | 7e-5 | 2.5e-2 |
| est | +0.0223 | | | 6e-17 | 4e-4 | |
| bh | +0.0142 | | | 8e-5 | 5e-5 | |
| mesh1024 | +0.0114 | -0.0004 | -0.0007 | 1e-8 | 7e-5 | 1e-4 |
| mesh2048 | +0.0025 | | | 5e-9 | 1e-5 | |

### Distribution and audit
- Radial quantile ratios (r75..r99 vs exact), max deviation over run: adaptive 3.0% flyby / 3.4% s12000 / 7.7% uniform; geo_audit 2.6 / 3.1 / 7.3; mesh1024 0.9 / 0.6 / 1.2. Uniform peak is at the collapse (~step 600).
- Pointwise density correlation vs exact dips mid-run for everything: adaptive min 0.39 (step 2750), geo_audit 0.64, est 0.86, bh 0.99, mesh1024 0.87, mesh2048 0.39 (step 2550). Seed 12000: adaptive 0.57, geo_audit 0.79, mesh1024 0.84. Uniform: all about 0.92-0.93 at step 600. A finer mesh dips as far as adaptive, so the dip is chaotic divergence around the core merger, not a method-quality signal.
- Noise floor (1e-6 perturbed exact) exists only to step 500 (dx_mean 7.3, corr 0.9985, quantiles within 0.1%). At step 500 adaptive: dx_mean 27, corr 0.996, quantiles within 1.6%; mesh2048 dx_mean 24.7, so pointwise position error is comparable across all methods by step 500. No comparison to the floor is possible after step 500.
- Audit: mean 0.0092 vs target 0.01, max 0.011-0.012, 19% of audits above target (geo_audit 23%, max 0.017). No fallback events in any run. Estimator fraction 1.0.
- Frame review (unbiased subagent, flyby + uniform grids exact/adaptive/mesh1024): rows visually indistinguishable at every step, no grid/stripe/block artifacts; flyby core merger between steps 1500-3000, uniform square collapses to a round core by step 2000.

## Verdicts vs predictions
1. adaptive |dE| <= geo_audit and <= mesh: FALSE. Adaptive is about equal to geo_audit (0.3% of E0 worse) and worse than both meshes (2.2% vs 1.1% / 0.25% flyby). Consistent with N=5000 smoke.
2. Momentum: tree methods 1e-17 (better than the 1e-12 predicted); bh 8e-5 as predicted; mesh 1e-8, much better than the 1e-4..1e-3 predicted.
3. Statistics within noise floor: cannot be tested past step 500 (floor run too short); by step 500 adaptive is outside the floor on quantiles (up to 1.6% vs 0.1%) but matches geo_audit/bh/mesh. Density correlation not distinguishable from finer mesh.
4. Cost: adaptive interactions 0.40x geo_audit on flyby and 0.56x on uniform (prediction 0.27-0.34x too optimistic; 1.0-1.25x uniform too pessimistic). Wall clock only 0.92x geo_audit on flyby and 1.11x on uniform, so the interaction saving barely shows in time. Mesh is 8x faster than adaptive.
5. No fallback events on any run, including the collapse. Not exercised, so H3 evidence is silent here.
6. Mesh1024 on collapse: FALSE, conserved well (P 2e-9, L 1e-4, dE within 0.07% of exact).
Additional: on the uniform collapse both tree methods lose angular momentum (1.5-2.5%) while mesh keeps 1e-4; and adaptive/est L drift 3-4e-4 on flyby is 5x geo_audit's.

