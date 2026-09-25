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
(pending; see status below)
