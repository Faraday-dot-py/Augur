# Dense-region scaling ideas for the 1B N-body sim (2026-09-25, discussion only, nothing implemented or tested)

## Problem
Near-field cost per body scales with local density x cutoff area, and each pair eval is the learned MLP (1-64-64-1, ~4k MACs). Uniform 1B at cutoff 4: 21.7 s/step, 67 GB (results at uniform density); uniform collapses by ~4000 steps, after which cores are far denser. Dense regions also concentrate into a few strips/tiles (imbalance, OOM risk).

## Constraints from the user
- Keep the sim generalizable: hesitant to tabulate the learned force as a lookup table (LUT bakes in a 1D central-force assumption; fine as a cache for CentralForceDynamics only).
- Do not give up accuracy except through learned models, if avoidable. Prefer a computation gradient by density over a regime switch.

## Options considered
1. Practical, physics-neutral: Morton/space-filling-curve sort every few steps (memory coherence); balance tiles by body count not strip width; block (individual) timesteps (small dt in cores, big in voids).
2. Density-independent per-body cost, with tradeoffs: mesh-only in dense cells (softened at cell scale, loses close encounters/relaxation; collision experiments showed merged cores are featureless virialized blobs, so may be fine); capped/sampled K neighbours (noisy, can break momentum unless symmetric); merge into weighted super-particles (loses identity); learned field model (needs training, generalization unknown).
3. Chosen direction (a continuous gradient): adaptive FMM.
   - Tree subdivides until each leaf holds <= K bodies, so dense regions just get a deeper tree; cost ~O(N) regardless of clustering.
   - Exact learned pair force only between adjacent leaves (bounded pairs per body); everything else via group interactions with a tolerance (expansion order / opening criterion) that converges to exact as tightened.
   - Kernel-independent FMM needs only evaluations of the kernel, so the learned MLP works without a LUT or analytic expansions.
   - Symmetric cell-cell interactions give exact momentum.
4. Alternative, smaller job: smooth force splitting f = f_short + f_long (Gaussian-smoothed long part) with a density-dependent split radius, symmetric per pair (e.g. min of the two radii), long part on a multi-level mesh with the same CIC deposit/gather (see mesh-momentum-investigation.md).
5. Where learning fits: a small learned corrector for the group-to-body far field beyond the monopole, derived from a learned potential so it stays conservative. Exact near field stays exact.

## Caveat
Any density-decoupled method approximates something; a gradient makes the error a tunable tolerance, not a fixed loss in one region. Adaptive FMM on GPU is the largest option (est. multiple days); mesh + smooth split is smaller.

## Suggested first step (not started)
Measure the pair-count distribution and per-tile time on a collapsed 1M clustered config, to see how much time the dense near field really takes before building the FMM.

Related: mesh-momentum-investigation.md (cic deposit/gather fix), lut-adaptive-substep-benchmark.md (in progress), experiment-log.md N-body sections.
