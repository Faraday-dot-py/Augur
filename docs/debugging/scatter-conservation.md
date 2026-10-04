# Scatter-field conservation / long-horizon (plan agent 6), 2026-10-04

Model: Exp E checkpoint (`E_ms_kp_pot_v_g128.pt`, 60769 it), inference flags kcache,fastio,cl,graph. Script `scripts/scatter_conservation.py` (+ `polaris_scatter_conservation.sh`), results in `results/conservation/`. Research only, no model changes.

## Setup

Regimes (scenes per seed x seeds 9100/9200/9300/4738): nr2 (8), nr10 (4), nr100 (1), nr300 (1): non-relativistic scale-init as Exp B (N=300 is beyond the 10-200 training range, spread +-30 against a +-32 grid); bh10 (4), bh100 (2), bh300 (1), bh1000 (1): relativistic c=10 cold Gaussian collapse as `scatter_bh.py` (sigma 8, vfac .3; scene 0 of bh300 = harness scene). 10000 ticks (1e3 is the prefix). Unit masses, dt .1, eps .5, gravity as the training sim.
Truth = exact softened all-pairs, fp64, KDK, 4 substeps/tick (training truth, "t4"); controls t1 (1 substep, the model's own 1-force-eval/tick integrator with the exact force) and t16 (converged) for N<=300. All metrics are computed per tick in fp64 from the (pos, vel) state, identically for model and truth: E=K+U (relativistic K=c^2(gamma-1)), |P|, L about the COM, COM, virial 2K/|U|, bound fraction (K_i+Phi_i<0), ejected (unbound and r>2 r_rms,0), fraction outside the +-32 grid, min pair distance, pairs closer than .25, max speed, max |dv|/dt, finiteness.

Integrator note: nonrel regimes use `model.step` (Verlet, force fed v+dt a_prev); BH regimes use the scatter_bh.py/harness KDK-on-momentum loop (force fed the half-kicked v(p)). Both as in the existing evaluation code.

## Predictions (written before any result)

1. Momentum: model |P| stays at float32 roundoff (momfix subtracts the mean acceleration, so sum m dv = 0 exactly): <=1e-3 at 1e4 ticks, not a failure mode. COM drift therefore tiny non-relativistically; relativistically COM moves as sum v(p) is not conserved, same as truth.
2. Angular momentum: truth conserves L to roundoff (central pair forces, KDK). Model does not (grid net with velocity input + recurrent hidden field is not rotation-equivariant or central; mesh lattice breaks isotropy): dL/L_scale grows with time, ~1e-2 by 1e3 ticks and O(0.1-1) by 1e4 in N=2 and N=10 (regular orbits make it visible), larger in absolute terms for N>=100.
3. Energy: truth t4 bounded, |dE|/(K+|U|) ~1e-3-1e-2 at 1e4 for N>=10 (close encounters), N=2 ~1e-4. Model: no energy structure; I expect systematic heating (E rises) in N>=10, |dE|/(K+|U|) ~0.05-0.3 at 1e3 and O(1) at 1e4; N=2 shows a secular orbit change (precession/spiral) so bound pairs unbind within 1e4 ticks in some seeds.
4. Finite grid (+-32): bodies outside get no mesh force and are not scattered, so they feel only the 2.0-radius pair term. Ejected bodies fly off and unbound-by-truncation (truth: bound orbits with apocentre beyond 32 would return). Prediction: model bound fraction decays faster than truth, outside-grid fraction grows with time in nr300 (starts at the edge) and nr100; far-field force at R>32 is zero in the probe.
5. |v|<c: guaranteed by construction (momentum state, v=p/sqrt(1+p^2/c^2)); expect 0 bodies with |v|>=c for model and truth; the informative quantity is the fraction above .9c (compare model with truth) -- model should have fewer or equal fast bodies because it does not resolve the deepest core.
6. Close encounters: softened force is bounded (max accel ~.38/eps^2 = 1.5 per unit mass), so no singular blow-up; I expect no non-finite state in 1e4 ticks, but large spurious |dv|/dt spikes (> truth max) at pair distance < ~.2 (pair MLP extrapolation, grid below one cell .5).
7. Collapse regime (bh): the model core settles with wrong energy: virial ratio drifts from truth after the collapse (tick ~50), and bound fraction/ejections differ; the late-time core is hotter and larger than truth (numerical heating at grid scale .5) -- direction untested guess.
8. Architectural flags I expect to confirm: (a) non-central/velocity-dependent/position-dependent force (probe: tangential component and Newton-3 residual of the 2-body force, lattice dependence), (b) no finite-grid far-field, (c) no symplectic/energy structure. Not expected: anything about non-antisymmetry in the near-field pair term (it is antisymmetric by construction for equal masses).

## Results (job 3323, H200, 10000 ticks, all 8 regimes, model + truth each < 30 s; smoke 3318; probe in `results/conservation/probe.json`; tables `results/conservation/summary.json`; onset script `scatter_conservation_onset.py`)

Scale for dE and dL: dE/(K0+|U0|), dL/(r_rms0 sqrt(2 N K0)) (L about COM). Median over scenes. "t1" = exact force with the model's 1-eval/tick integrator; t4 = training truth; t16 = converged.

| regime (B scenes) | |dE|/scale m @100 / 1e3 / 1e4 | truth t4 @1e4 (t1 @1e4) | |dL|/Lscale m @1e3 / 1e4 | t4 @1e4 | bound frac m / t4 @1e3 | bound m / t4 @1e4 | first tick model dE>0.1 | model non-finite | model |v|>=c |
|---|---|---|---|---|---|---|---|---|---|
| nr2 (32) | .016 / .11 / .61 | 6e-7 (1e-5) | .33 / 3.5 | 0 | 1 / 1 | 0 / 1 | 730 | none | n/a |
| nr10 (16) | .018 / .094 / .91 | 3e-5 (9e-3) | .29 / 1.8 | 0 | 1 / 1 | .5 / .8 | 1175 | none | n/a |
| nr100 (4) | .0035 / .020 / .39 | 2e-4 (.31) | .040 / .34 | 0 | .905 / .975 | .39 / .715 | 3700 | none | n/a |
| nr300 (4) | .031 / .25 / 1.35 | 4e-4 (.43) | .072 / .27 | 0 | .72 / .955 | .012 / .77 | 300 | none | n/a |
| bh10 (16) | .011 / .15 / 3.0 | 7e-5 (8e-3) | .68 / 4.2 | 1e-6 | .95 / 1 | .30 / .80 | 670 | none | never |
| bh100 (8) | .009 / .17 / 2.1 | 3e-4 (.53) | .17 / 3.5 | 4e-5 | .88 / .935 | .02 / .645 | 725 | none | never |
| bh300 (4) | .035 / .45 / 2.0 | 1e-3 (.83) | 3.1 / 5.6 | 3e-4 | .76 / .885 | .007 / .61 | 260 | none | never |
| bh1000 (4) | .025 / .84 / 1.9 | 4e-3 (n/a) | 5.3 / 9.4 | 2e-3 | .51 / .86 | .007 / .585 | 495 | none | never |

(dE sign: model E rises at 1e4 in all regimes; at 1e3 it falls in the bh100/bh300 collapse regimes, -.17/-.45, then rises.)

Other measurements:
- Momentum |dP| (model): 1e-6 (N=2..10) to 1e-3 (bh300) to 3e-2 (bh1000) at 1e4 ticks, exactly 0 for nr2; float32 round-off, never a failure. COM drift model non-rel: 3e-5 (N=2), 5e-3 (N=300) at 1e4 (truth 0): fine.
- |v|<c: held in every model scene (model vmax/c .43-.99, truth .33-.99; fraction of bodies above .9c at peak bh1000 model .51 vs truth .52, bh300 .06 vs .07). Zero violations, as constructed (momentum state). Not an issue.
- Non-finite: none in any of 8 regimes x 10000 ticks. Min pair distance (softened, eps .5): model 9e-6..8e-3 vs truth 2e-5..1e-3, no blowup. Max |dv|/dt: model 1.5-1.6x truth (bh1000 168 vs 166, bh300 109 vs 82, bh100 52 vs 33, nr2 4.8 vs 1.5), bounded; close encounters give larger but finite kicks.
- Virial 2K/|U| at 1e4: model 4-870 (nr2 6, nr10 4, bh10 10, nr100 1.4, nr300 127, bh100 135, bh300 411, bh1000 872) vs truth .7-.85: model is a hot unbound gas, truth stays virialised (0.7-0.85) with ~60% bound.
- Grid: truth also leaves the +-32 grid by 1e4 (outside fraction .97-1 for expanding scenes), so scenes past ~1e3 ticks are largely outside the model's domain; this confounds the late conservation numbers with the finite-grid cutoff. Onset of model-vs-truth divergence is earlier than the exit: bound fraction falls to half of truth's at tick 1425 (bh1000), 2175 (nr300), 2850 (bh300), 5175 (bh100), 4875 (nr2), 7350 (bh10), never for nr10/nr100; at those times outside-grid fraction is already .1-.4 for N>=100 but 0 for N<=10 (nr2, nr10, bh10 never leave the grid in 1e4 ticks, yet still heat).

Probe (probe.json; 64 random placements, v=0 unless noted, momfix on):
| r | radial force / true (median) | tangential/true rms (torque) | |F(v)-F(0)|/true | rel err rms |
|---|---|---|---|
| .1 | .79 | .008 | .007 | .21 |
| .5 | 1.00 | .028 | .013 | .06 |
| 1 | .98 | .033 | .028 | .06 |
| 2 | .85 | .051 | .096 | .19 |
| 4 | 1.03 (std .81) | .16 | .26 | .84 |
| 8 | 1.10 (std .70) | .42 | 1.0 | .84 |
| 16 | 1.47 (std 2.6) | 1.2 | 3.9 | 3.0 |
| 30 | 2.5 (std 9.0) | 3.6 | 14 | 10 |
Newton-3 residual (sum of the two accelerations) 1e-8 at every r (momfix). Curl/total of the force of a static star .13 (rms curl .010, rms div .076). Far field of a 100-body cluster on a test token (true accel in x): ratio .95 at R=5, 1.03 at 10, 1.01 at 20, 1.25 at 28, 4.0 at 31 (grid-edge artefact), .006/.008/.019 at R=33/40/60 (force cut off; residual 5e-4 is the pair-term/momfix leftover); constant spurious y-acceleration .017 beyond the grid (momfix redistributing the cluster's net force, see below).

## Findings

1. No catastrophic failure: no NaN/inf, no |v|>=c, no pair-distance blow-up, momentum at roundoff, COM stable, in 10000 ticks at N up to 1000. The model is numerically stable; it is physically wrong on the conserved quantities that are not hard-wired.
2. Energy: the model gains energy secularly, at a rate far above both the training truth (t4: <=4e-3 at 1e4) and the same integrator with the exact force (t1: 1e-5 for N=2 to .8 for BH). Time to 10% error 260-1175 ticks (26-118 time units) at N<=300 except nr100 (3700); at 1e4 ticks the energy error is O(1)-O(3) of the initial total in every regime. Model heating exceeds t1 by 10-1000x at low N and 2-10x at N>=100 (bh300 .35 vs .016 @100, .45 vs .23 @1e3; nr100 .02 vs .0095 @1e3). Hence about half (N>=100, early) to ~all (N<=10) of the energy drift is learned-force error rather than the 1-eval integrator.
3. Angular momentum: the truth conserves L to roundoff (0 to 1e-6 relative at 1e4); the model loses/gains it at .02-.03 per 100 ticks of the orbit scale in N=2 (|dL|/Lscale .33 at 1e3) and breaks it entirely by 1e4 (3.5 = pair unbound). Cause is measured: the force on a pair is not central (tangential component 3% of the true force at r=1, 5% at r=2, 16% at r=4, 42% at r=8) and depends on the bodies' velocities (3% at r=1, 10% at r=2, 100% at r=8, with N=2 and random v) and on placement (radial ratio std .05 at r=1 for random positions). The mesh (CIC onto 128^2 grid, 0.5 cell) breaks isotropy; the potential net and velocity-fed inputs break centrality.
4. Where it fails by regime: (a) N=2/10 (bound regular orbits): slow secular heating, precession, pair unbinds around tick 4000-7000 (bound fraction 0 by 1e4 for nr2; .5 for nr10), no domain issue; (b) N>=100 and BH collapse: heating from tick ~300-700, plus the grid cutoff: bodies beyond +-32 receive no mesh force (probe: force x.006 at R>=33) while momfix adds back a uniform spurious acceleration, so ejecta from the model fly off ballistically: bound fraction collapses to <.02 at 1e4 vs .6-.8 in truth, virial 100-900. (c) Collapse (BH) first 100 ticks: dE -.035 at tick 100 (bh300) with the exact-integrator error only +.016: the model under-bounds the collapsing core (negative dE = energy removed, then heated later).
5. |v|<c cap: not a failure mode (construction). The model matches truth's fast-body fraction even at N=1000 (.51 vs .52).
6. Chaos caveat respected: judged on E, L, bound fraction, virial, not per-body positions; truth t1/t4/t16 spread (t4 vs t16: |dE| 3e-4 vs 1e-5 at bh300 1e4) is 2-4 orders below the model deviations except at nr2 early (7e-7 vs 1.6e-2 at 100: still 4 orders).

### Architectural vs. not
- Architectural: (i) force is not a pairwise central force (curl .13, tangential 3-40%, velocity dependence): energy and angular momentum cannot be conserved, only momentum is enforced (momfix); (ii) finite grid with zero force outside +-32 and a ~x4 edge artefact at 31: any scene that expands past the box loses its far field; momfix then injects a uniform spurious acceleration that grows with the unbalanced force; (iii) no energy/symplectic structure; the Verlet integrator is only 2nd-order symplectic if the force is conservative, which it is not.
- Not architectural / training-limited: the near field (r<=1: radial within 2% median, though .79-.85 at r=.1 and 2); error at r>=4 grows (rel err .84 at r=4-8) partly because the training distribution (Exp E, N 10-200) rarely samples isolated far pairs at that precision (untested); the 1-eval Verlet integrator contributes but is not dominant.

### Caveats
- 1 to 4 scenes per seed (bh300/nr100/nr300/bh1000: 4 scenes total), medians over few scenes; 1e4 scenes share the confound that truth also leaves the grid.
- Probe is v=0 with random placement; the "velocity dependence" row uses random v of rms 2; the N=2 probe is a 2-body environment, the cluster far-field probe uses 100 bodies of sigma 3.
- Truth substeps differ for t1 (1 substep at dt .1, rather than the training truth).
- Review subagent output: see below.

## Unbiased review (fresh subagent, prompt docs/debugging/scatter-conservation-review-prompt.md, grids `videos/scatter_conservation_<regime>_{ts,snap}.png`)
Summary of what it reported: all regimes track truth to t~100 (bh300 to ~30-100, bh1000 ~50-100, nr300 diverges from ~100-200). Then model signed dE rises smoothly (N<=10: to +.9..+3 at 1e4) or, in the collapse regimes bh100/bh300, first goes negative (-.35 near 2e3, -.5 near 1e3) and then jumps positive. N<=10 (nr2, nr10, bh10): model stays clustered near the origin (median radius flat ~20, outside-grid fraction 0) while truth spreads to hundreds of units; in nr2 both bodies are off-panel by t=1e4 while truth stays within ~5. N>=100: model median radius 2000-9000 at 1e4 vs truth 100-200, bound fraction ~0 vs .6-.7, virial 100-900 vs ~1, COM drift 30-700 vs 2-20, ejected ~1 vs .3-.4; max |dv|/dt collapses after ~5e3 in the model (everything far apart), truth keeps ~150. dL erratic, signs differ by regime.

Reconciliation / new observations from the review:
- Two distinct late behaviours: N<=10 model bodies stay near the origin (hot, confined; virial 4-10, bound .3-.5) while truth ejects to hundreds of units; N>=100 bodies are expelled to thousands of units. The confinement at low N is consistent with the spurious inward force at the grid edge (probe: x4 true attraction at R=31, then ~0 beyond R=33), i.e. a soft wall at the grid boundary; bodies cross R~32 only if energetic enough. Untested directly (would need trajectories reaching R=31 at N<=10; model nr10/bh10 never exceed grid in 1e4 ticks, outside-grid 0, which supports it).
- The collapse-regime dE dip (negative before tick ~1e3) is the model's under-bound core relaxing with too little energy before heating dominates.

## Ranked fix candidates (not implemented; research only)
1. Make the force conservative and central: have the net output only a scalar potential (already `pot`) and drop the velocity input to the force and the recurrent hidden field for gravity, or add an explicit pairwise antisymmetric central near-field plus learned smooth mesh potential; evidence: tangential 3-40%, velocity dependence 3-100%, curl .13, energy +O(1) and dL breakdown at 1e4. Expected: dL and dE drift near the t1-integrator floor (1e-5 to 1e-2). Cost: retrain; medium risk to accuracy.
2. Fix the far field / domain edge: analytic or learned 1/r monopole (+quadrupole) outside the grid (or periodic/zero-padded potential with a far-field correction), taper the grid-edge artefact (x4 at R=31, 0 beyond 33). Evidence: probe far-field, bound fraction collapse at N>=100, N<=10 confinement. Expected: removes the expulsion/virial 100-900 failure and extends valid horizon past ~1e3 ticks for expanding scenes. Cost: low (inference-side addition, may need short fine-tune).
3. Energy-aware training signal: add dE/dL and force-curl penalties on long unrolls (rollout 100+ ticks, not 30), or train with the two-body/cluster long-horizon windows (training windows are 30 ticks, error grows beyond). Cheap, complementary to 1; prior note (scatter-field-overnight-notes iter10) shows long-horizon training halved conservation error.
4. Integrator-side: 1-force-eval Verlet (t1) already costs +.016 (bh300 @100) to +.8 (@1e4) of energy; 2 substeps/tick with the model force (cost 2x) or a slower dt would cut that component; benefit bounded since the learned-force error is 2-1000x larger.
5. Diagnostics to keep (cheap): per-run |v|<c fraction, bound fraction, virial vs truth as a standing gate in scatter_regress.py (not done).
Decisions for the user: whether conservation of E and L (not just momentum) is a hard requirement for the paper; fix 1 is a retrain, fix 2 is not.
