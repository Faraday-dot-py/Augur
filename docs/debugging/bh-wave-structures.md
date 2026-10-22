# BH collapse: wave-like structures and spin (PRE-REGISTRATION, written before any run)

Date 2026-10-22, branch `bh-wave-structures` (from `exp-e-proposed-1e8`). Seed 4738. Analysis scripts only; model untouched.

Scene (from `scatter_bh.py` / `scatter_bh_1b.py`, read before registering): N unit-mass bodies, Gaussian sigma 8 clipped to +-29, per-component IC speed std 0.3*sqrt(N/(4*sigma)), then clamped to 0.99c; c = 10, dt = 0.1, softened force eps = 0.5, G = 1, relativistic momentum state p (v = p/sqrt(1+p^2/c^2)), grid 128^2 on extent 64 (cell h = 0.5 = eps), mesh-only (pair term off). Horizon R_s = 2N/c^2.

## Hypotheses and confirm / reject criteria

| id | hypothesis | CONFIRMED if | REJECTED if |
|---|---|---|---|
| H1 | Physical: orbit-crossing / shell-crossing of a near-radial, speed-capped collapse (bodies pass through the core at ~c and fly out as shells; radial oscillation, phase wrapping) | exact (truth) sim at small N shows the same shells with the same radii/time; 1-D spherical-shell integration with the same softening, cap and mass profile reproduces shell radii vs time within ~15% | no structure in truth, or reference radii/period disagree |
| H2 | Speed-cap bounce: shell front travels at c (1 unit/tick) | shell front radius slope vs tick = c*dt = 1.0 +-10% in truth and model | slope differs |
| H3 | Mesh artefact: structure scale locked to grid cell h = 0.5, box +-32, FFT periodicity | structure radii/wavelength change when the grid or extent changes at fixed N, or plain PM at same grid shows it while exact truth does not, or radii cluster at multiples of h | radii independent of grid |
| H4 | Learned-model error (U-Net potential term) | model shows structure that plain PM (non-learned) at same grid does not, or model structure differs from truth where PM matches | PM equals model |
| H5 | Shot noise: structure amplitude drops with N (1e4 -> 1e5 -> 1e6) | azimuthal power in rings relative to Poisson floor falls as N grows | ring contrast N-independent (converges to a physical limit) |
| H6 | Edge: reflection/pile-up at the +-32 box (zero force outside) | rings change at r ~ 29-32 or time shift matches round trip to the edge | no effect of edge on ring radii at r < 25 |
| Q4 | Spin: IC has net L = 0 (isotropic random velocities, momentum subtracted), so no net spin-up; rotation rate stays ~0 in truth and model. A rotating-IC variant should spin up as 1/r^2 (L conserved) in truth; model conserves L only if the mesh force is central (momfix removes net force only) | L(t)/L_scale < 1e-2 truth; rotating IC: L drift model vs truth <5% | L grows from nothing |

Quantities: radial density profile rho(r, t), shell radius vs tick (local maxima of rho(r)), azimuthally averaged 2-D power / ring contrast, energy E(t), virial ratio 2K/|W|, L(t) about the centre of mass, rotation rate Omega(<r) = <L_z>/<r^2> within enclosed-mass radii.

Constraint check: generalization, O(N) and interpretability untouched (analysis only); pp=2 off as in the 1e8 runs; truth E(t) is the reference, not E(0).
