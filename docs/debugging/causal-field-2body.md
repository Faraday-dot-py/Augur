# Causal field medium: 2-body prototype

Status: in progress, uncommitted. Script `scripts/causal_field_2body.py`, job script `scripts/polaris_causal_field_2body.sh`. Seed 4738 (no randomness in the physics; seed only set).

## Thesis under test
Instantaneous all-pairs attention violates finite speed of information. Information should propagate through a medium at speed <= c: influence from distance r arrives after ~r/(c dt) steps; a local field state updated by local stencils gives O(N) per step. Delay alone does not beat N^2 (recompute every k steps = N^2/k); the gain would come from the field state carrying information hop by hop.

## Setup (all on Polaris GPU, fp64)
Units G=1, softening eps=0.5 (matches gravity_sim / CentralForce training), star mass M=1 at origin, planet mass m (m=0: test particle), circular orbit radius R, c=3 (v/c ~ 0.1 at R=10), dt h=0.05, KDK leapfrog, deletion at t0=30.
Variants:
- instant: analytic Newton, all pairs, same time.
- retarded: Newton force evaluated at the source's retarded position/time, solved by 8 fixed-point iterations of t_r = t - |x_target(t) - x_src(t_r)|/c with linear interpolation of a history buffer (positions before t=0 assumed stationary at the initial position). The deleted star contributes only if t_r <= t0. This is NOT a conservative theory (no velocity-dependent terms).
- token: existing CentralForceDynamics checkpoint `checkpoints/gravity_central_v1.pt` (instantaneous pair force, unit masses), star fixed, planet test particle.
- field: 3D scalar wave eq `phi_tt = c^2 (lap phi - 4 pi rho)`, 7-point Laplacian, leapfrog in time, dx=0.5, CFL number kappa = c dt_f/dx (stable for kappa <= 1/sqrt(3)=0.577). One field per source body (no self-force; costs one grid per body, so not O(N) as is). Source deposited bilinearly on the z=0 plane, planet reads central-difference gradient bilinearly interpolated, a = -grad phi. Outer shell Dirichlet = exact monopole -m/r with retarded switch-off (so no wall reflection in the deletion test; for moving bodies it is a monopole about the COM, approximate). Field warmed up with bodies pinned for 3 L/c before release.
Note: for a moving source, -grad(retarded scalar potential) differs from "Newton at retarded position" at O(v/c); the two are different theories, so field vs retarded truth are not expected to agree for m>0.

## Predictions (written before any run)
Delete-the-star (R=10, c=3, expected R/c=3.33):
| variant | predicted onset latency (50% drop) | predicted onset shape |
|---|---|---|
| instant | 1 step (0.05), i.e. t0 | step to 0 |
| retarded | exactly R/c = 3.33 (+-1 step), for the circular orbit the distance stays R so no correction | step to 0 |
| token (learned) | 1 step; model sees only present tokens | step to 0 |
| field | ~R/c within ~2-3 grid-crossing times (front smeared over ~ a few dx/c = ~0.5-1), possibly early by dispersion | smeared drop, plus a transient spike in |a| at the front (a source switched off leaves a 1/r -> 0 step in phi, gradient is a delta-like shell); pre-front a unchanged |
Latency vs R: linear with slope 1/c for retarded and field, flat ~0 for instant/token. Latency vs c at R=10: ~1/c.
Orbits (R=10, 10 orbits, period ~199):
- m=0 (heavy static star): retarded identical to instant (static source) to machine precision; both conserve E to leapfrog-level (<1e-3 relative, bounded). Field: not exactly conservative; expect r oscillation/precession from grid anisotropy + effective softening ~ dx, dE/E ~1e-2 or smaller; no secular blow-up.
- m=0.01/0.1/1 retarded: secular energy GAIN (orbit expands), my hand estimate: tangential accel fraction ~ v_p/(2c); per-orbit dE/E ~ 0.1-0.3 for equal masses at c=3, ~ proportional to m/(M+m) roughly for smaller m, ~2x larger at c=1.5. Low confidence on magnitude, moderate on sign and on "equal mass is worst".
- field m>0: bounded but different from retarded; E drift O(1e-2)/orbit, sign unknown (not predicted).
- token: tracks analytic central orbit to the learned-force error; a_ratio near 1 within a few %.
CFL: stable for kappa <= 0.577, blow-up (exponential growth, nan) for kappa >= 0.6; kappa=0.55 marginally stable. Smaller kappa (0.1) gives more numerical dispersion: front arrives later/blurrier? Prediction: onset error grows as dx grows (dx=1.0 worst); dx=0.25 closest to R/c; static accel at dx=1 within ~1% of analytic at R=10.

## Results
(to be filled)
Jobs: smoke 3192 (crashed: OOB index after blow-up, plus 20-30% pre-deletion noise from grid ringing), smoke2 3193, full run 3195 (all sections, 3 min GPU). Fixes between: damped relaxation warm-up (gamma 0.03) so the field starts on the discrete static solution, index clamp. Data: `results/causal_field_2body.json`, `results/causal_field_2body_ts.npz`. Plots `videos/causal_field_2body_{accel,latency,orbit,cfl}.png`.

### Delete the star (R=10, c=3, R/c=3.33; latencies in time units, 50% drop of |a|)
| variant | predicted | measured lat50 | notes |
|---|---|---|---|
| instant | 0.05 | 0.05 at all R, c | as predicted |
| retarded | R/c (+-1 step) | R=3..40: 1.05, 1.70, 2.55, 3.35, 5.05, 6.70, 10.05, 13.35 = R/c + 0.05 exactly; c sweep at R=10: 6.7, 3.35, 1.7, 0.85 = R/c + 0.05 | matches to 1 step |
| token (learned) | 0.05 | 0.05 for R=5,10,20,30 | learned force a_ratio vs analytic at R=10: 1.0018. Reacts at t0 (cannot do otherwise: no history, no field state) |
| field (dx=0.5, kappa=0.15) | ~R/c, smeared | R=5/10/20/30: 1.30/2.95/6.05/9.25 vs 1.67/3.33/6.67/10.0 (about 0.4-0.75 EARLY, 11-22%); c=1.5: 5.8 (6.67), c=6: 1.5 (1.67) | scales ~linearly in R and 1/c, so the medium genuinely carries the delay; but threshold crossing is early and the |a| spike (peak 7.7x a0 at R=10, growing with R: 3.5, 7.7, 12, 15.5) peaks LATE at t_peak 3.5 (R=10), 1.75 (R=5), 6.9, 10.3 |
So the field result partly disagrees with the prediction: onset is early by 0.2-0.7 depending on threshold, not within a smear of +-0.5 around R/c in a clean way; the front is front-loaded (a 5% deviation appears at 2.7, 50% drop at 2.95, spike at 3.5). I did not isolate whether the early part is the grid's numerical domain of dependence (stencil speed up to dx/dt_f = c/kappa, precursors outrun c) or the CIC source/stencil footprint (about 2 cells = 0.25-0.4 time units at dx=0.5); the dx sweep (below) is consistent with a footprint/precursor effect that shrinks with dx. UNTESTED.

### CFL / resolution (R=10 delete test)
| kappa | h | outcome | lat50 | peak/a0 |
|---|---|---|---|---|
| 0.1 | 0.017 | stable | 2.95 | 7.7 |
| 0.3 | 0.05 | stable | 2.90 | 7.9 |
| 0.5 | 0.083 | stable | 3.00 | 8.5 |
| 0.55 | 0.092 | stable | 3.00 | 8.6 |
| 0.6 | 0.1 | BLEW UP (max phi 1e118) | - | - |
| 0.7 | 0.117 | BLEW UP | - | - |
Instability threshold sits between 0.55 and 0.6, matching the 3D leapfrog bound 1/sqrt(3)=0.577. Latency is nearly kappa-independent in the stable range (smaller kappa not more accurate here; dispersion error is set by dx).
dx sweep (kappa 0.15-0.2, nsub 1-3): dx=1.0/0.5/0.25 -> lat50 2.70/2.95/3.05 (error -0.63/-0.38/-0.28 vs 3.33), static accel vs analytic at R=10: x1.013/1.008/1.006. Error shrinks slowly with dx (about 40% per halving, not 4x), the peak overshoot grows (4.4/7.7/13.2) as the delta-shell front gets sharper.

### Orbits (R=10, 10 orbits, period ~199 for M+m=1; E and L from instantaneous Plummer PE)
| variant | m | dE/|E| final (max abs) | dL/L final | separation range |
|---|---|---|---|---|
| instant | 0, 0.01, 0.1, 1 | 0.0 (to 5 digits) | 0.0 | 10.0-10.00002 |
| retarded, c=3 | 0 | 0.0 (static star => identical to instant) | 0 | 10.0 |
| retarded, c=3 | 0.01 | +0.19 | +0.11 | 10 -> 12.3 |
| retarded, c=3 | 0.1 | +0.59 | +0.57 | 10 -> 25 |
| retarded, c=3 | 1.0 (14 orbits) | +0.82 | +1.28 | 10 -> 65 |
| retarded, c=1.5 | 0.1 / 1.0 | +0.70 / +0.90 | +0.83 / +1.63 | 10 -> 33 / 115 |
| field, c=3 | 0 | +1e-5 (max 1.7e-4) | 1e-5 | 9.88-10.0 |
| field, c=3 | 0.1 | -1.1e-3 (max 5e-3) | -1.3e-4 | 9.71-10.25 |
| field, c=3 | 1.0 | +2.0e-2 (max 4.9e-2) | +1.0e-2 | 9.99-10.39 |
| token (test particle, fixed star) | 0 | not computed; dL/L 1.7e-5 | | 9.99987-10.00067 |
Retarded-Newton prediction scorecard: sign (energy gain / orbit expansion) CORRECT; equal mass worst CORRECT in absolute terms but my "proportional to m/(M+m)" scaling was WRONG: for any m>0 the relative drift per orbit is similar (1.9%/orbit at m=0.01, 5.6% at 0.1, 5.8% at 1.0; magnitude lower than the 10-30% I predicted for equal mass), because the star becomes a test particle in the planet's lagged field. The retarded-Newton truth is secularly unstable (ejection within ~10 orbits) at v/c ~ 0.1, as Laplace's aberration problem says. The field model (a scalar wave theory, i.e. a different, better-behaved theory) stays bound over the same 10 orbits with <=5% energy wander, so in this setup the field is a more stable "causal" model than the naive retarded-force truth. Caveat: field vs retarded Newton are different theories for m>0, so "field error vs truth" is not a meaningful metric there; a physically consistent truth (Lienard-Wiechert-type) was not built. Field energy here also is not a conserved quantity of the particle system alone (energy is exchanged with the field).

### Cost
Field run: 3D grid 129^3-153^3 fp64, ~38-62 s per 10 orbits (40k steps x nsub 2), i.e. ~1 ms/step per grid; one grid per body, so this prototype is O(grid) per step per source body, not O(N). Not benchmarked at N>2 (untested).

### Caveats / negatives
- Planet test mass m=0 energy numbers use planet mass 1 substituted only in E/L bookkeeping (P_max for m=0 is an artifact of that, ignore).
- Retarded truth assumes pre-t=0 stationary history (small initial transient for m>0).
- Outer boundary Dirichlet monopole about the COM: exact for the static star and deletion test, approximate for moving bodies.
- Single seed/IC (deterministic), one c per section; no 3D orbits out of plane; no learned field.

### Unbiased plot review (fresh subagent, no hypothesis given, read the four PNGs only)
- accel: instant/token step to 0 at x=0; retarded steps to 0 at x=1 (R/c); field is flat to x~0.8, dips to ~0 at x~0.9 (before R/c), spikes to ~7.7x at x~1.05, then RINGS (3.8 at 1.2, 2.7 at 1.3, ~2 at 1.5-2.6, ~1.9 at 3.2) and has not settled to 0 (~0.2-0.3) by x=3.5-4. Looks like a grid-scale artifact.
- latency: retarded on R/c line; field parallel but 0.3-0.8 early (gap grows with R); instant/token at the one-step floor.
- orbit: retarded m=1 energy +0.8 and separation 10->65; field/instant stay ~10 with small wobble.
- cfl: curves for kappa 0.6/0.7 are not visible (they blew up, plotted as inf/NaN, so not distinguishable from the plot alone); ringing amplitude grows as dx shrinks; spikes clipped at y=3.
Takeaway added by me: the unfiltered delta-shell front is the main defect of the hand-coded field. The planet's force stays contaminated (|a| up to 2x pre-deletion) for >= 2 R/c after the front passes, where the exact answer is 0. Source-removal of a non-conserved charge is an abrupt event; a smoother source ramp or a filtered stencil would be needed before using this as a training target. Untested.
