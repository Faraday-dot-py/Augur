# Hole 1 close-out (branch hole1-potential-field, 2026-10-24/25)
Tested on 2D Lennard-Jones (LJ) scatter field, tier 0 only, seed 4738, held-out eval seeds 4739/4740. Numbers: docs/debugging/experiment-log.md (2026-10-24 hole-1 entries).

## Settled
- Drift cause: velocity and recurrent-field inputs to the mesh. Scratch with those inputs (S2) drifts dE/N -0.11 (solid), +0.20 (gas), +0.16 (coexist); the same non-gradient mesh with them zeroed (V) is <= 0.013; the potential-gradient mesh (P) is <= 0.0084. The original fine-tuned -0.26/-0.29 is therefore not a fine-tuning artifact.
- Gradient structure helps modestly, not decisively: P drift 3-10x below V; err@20 equal to 1.3x better. Residual force channel (PR) stays unused under its soft penalty (~1e-8).
- Reciprocity does not need to be built in: a pair term with no built-in reciprocity plus soft reciprocity/momentum penalties (A) learns reciprocal forces (penalty ~1e-6), drift <= 0.0071, solid err@20 1.08x P. Stable bonded block (rest and sticking collision) with P, A, V and M: Rg within 0.3%, nothing evaporates, momentum drift ~1e-7, dE/N <= 0.017; frame review found no artifacts. Limits: one geometry, cold, 160 frames.
- The solid err@20 gap (9x vs LJ v2) was integrator time step, not structure: exact LJ force at the model dt 0.025 already gives 0.0133 (P 0.0171). Retraining at dt 0.0125 (P2) gives 0.0029 at 1.8x cost with drift 0.0025; dt 0.00625 (P4) 0.0025 at 3.4x. Inference-only dt halving gets only 0.0114 (model learned a dt-specific compensation). Pair-only (RESPA) substeps match whole-model substeps in accuracy at inference but cost 1.9x as implemented (no neighbour cache).
- Mesh contribution on LJ is small: pair-only (M) is stable and low-drift (<= 0.0082); mesh helps solid ~12%, liquid/coexist ~1.5x; in P the long-range FFT kernel contributes ~0. The learned pair curve error is concentrated at r 0.9-1.2 (stiff core).
- Polynomial-cell potential (tier >= 1) is exactly conservative; the naive force blend is not; accuracy cost vs naive 0.9-2.5x (inconclusive, noisy fit).

## Open
- Whether the potential-gradient form pays off where the mesh matters (gravity, long-range, tier >= 1 networks): LJ barely uses the mesh. Tier >= 1 use of the polynomial potential is untested inside a network.
- Exact LJ pair term was unstable (quench blow-up, coexist drift 0.026-0.034): the stiff core needs the smaller dt or the learned softer curve; not resolved which.
- Remaining liquid/coexist err@20 (0.13-0.14, 0.045) is learned-force error plus chaos; pair-curve accuracy in the core is the lever, not dt.
- dt study is one training seed per dt; cost for pair-only substepping with a cached neighbour list untested; P2 not yet run through the full six-scene gate (T/psi6/P) beyond err@20/drift.
- Fine-tuning from an existing checkpoint at smaller dt, and non-reciprocal / velocity-dependent / dissipative cases (residual channel) were not tested; block test has no fragmentation or restitution case.
