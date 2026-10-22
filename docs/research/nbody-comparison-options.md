# Comparing Augur against other N-body simulators: options (research + proposal)

Date 2026-10-22. Research only: no model code changed, no GPU job run. Web sources fetched 2026-10-22; anything not read in full text is marked UNVERIFIED. Acronyms are defined at first use. "Augur" = the learned scatter-field model (stock Exp E, checkpoint `E_ms_kp_pot_v_g128.pt`).

## 1. What we have (so proposals reuse it)

### 1.1 Model and what it simulates (verified in repo docs/code)
- Physics being emulated: softened gravity with acceleration a = G m r / (r^2 + eps^2)^(3/2), G = 1, unit mass per body (the 3D-form Plummer law evaluated on 2D positions; `scripts/gravity_sim.py` `accel`, `rollout_torch`; `scatter_far_kernel_check` states the same law). So Augur is a planar system with 3D-law Plummer softening, not 2D log-potential gravity. Any 3D code can reproduce the truth by setting z = 0, v_z = 0 (planar symmetry keeps z = 0). A separate 3D model (`CentralForceDynamics`, `--dim 3`) exists (100k-body BH collapse, job 3151) but is not the scatter field.
- Units: G = 1, eps = 0.5, dt = 0.1, truth = fp64 all-pairs KDK (kick-drift-kick) with 4 substeps per tick ("t4"), exact reference chunked on GPU (`ops.exact_rollout`). Model = 1 force evaluation per tick.
- Architecture facts that bound any comparison: mesh grid 128, extent 64, cell h = 0.5 (V-shaped optimum in h; h = 0.25 is 25x worse, h = 1 is 5x worse); learned far kernel `kmlp` is right to r ~ 32-40 and wrong beyond 64 (R1b retrain to r 1500 shipped on branch `far-kernel-fix` with per-interaction retrain); pair term pp = 2.0 and knn = 16 for r < 2 (dense N^2, cell-list variant exists; stock cell-list collapses under cold collapse); zero force outside the +-32 box unless tiled; momentum fixed exactly (momfix), no energy or angular-momentum structure.
- Tiling: 32-unit cores + 16 halo = 64-unit/128-cell copies, ghost bodies + coarse global far term PASS vs single grid (err@20 ratio 0.95-1.00, force seam/interior 0.54-1.06; `scatter-tiling.md` on branches). 8x8 tiles N = 1200: err@20 .115 vs exact, .014 vs single (job 3785). Load balancing partly done (makespan -21..-45%).
- Zero-shot validity (Exp E): N <= ~1e3 cold collapse / ~5.6e3 warm per 64-unit box; scene width sigma <= ~16; core density sigma >= 5; mass ratio <= 3:1; eps = 0.5 only; dt 0.25x-2x of 0.1 (`scatter-scaling.md`).

### 1.2 Scenes and IC generators that already exist
| scene | generator / script | status |
|---|---|---|
| two-body, expB (10-100 bodies, scale-init) | `gravity_sim.make_dataset`, `scatter_regress.py` | frozen baseline `results/scatter_baseline_v2.json` |
| cold Gaussian / BH collapse (relativistic KDK, c = 10) | `scatter_bh.py`, `scatter_scaling.py` | harness bh300; N sweep 2..1e5 |
| binary orbit, planetary system, star+planets | `scatter_scenes.py` | binary scratch 7.9% period drift, planetary fails |
| globular = projected Plummer, virial q = .5, N = 100/300/1000, 1000 ticks | `scatter_scenes.py` | r50 band .85-.97, dE/E 3-18%, escapers 3x truth |
| disc galaxy (exp. disc Rd 4, 2D-Plummer bulge a 1.5, Toomre Q 1.5, no dark matter) | `galaxy_ic.py`, `scatter_galaxy.py` (unmerged galaxy branch, commits 08652c3..3b38f3f) | g1 (M 300), g2 (M 100); 1e9 runs done |
| solar disc / protoplanets | `scatter_solar.py` | abandoned by user at 1e7 |
| clumpy 1e9 BH, 1e9 BH | `scatter_bh_1b.py` | timing only |
| 3D BH collapse 100k | `orbit_bh.py --dim 3` | stable diagnostics, no reference at scale |
| dual-tree/FMM-like adaptive far field (analytic 100k states) | `dual_tree.py`, `dual_estimator.py`, `kernels.py` | unoptimised torch, wall time not comparable |

### 1.3 Error, conservation and cost metrics already defined (reuse verbatim)
- err@k = mean over bodies (then scenes) of ||pos_model - pos_true|| after k ticks, sim units (err@20 = 2.0 time units). Trivial baseline: constant velocity. Chaos / reference-noise floors per scene group: `results/chaos_floors.json` (`scatter-chaos-floors.md`); BH err@100 not quoted, headline err@50.
- Force error: acceleration-weighted relative error sum|a_model - a_exact| / sum|a_exact|, by radius tercile, core vs halo, with pair term ablated (`scatter_scaling`, tiling docs). Not percentile-based yet.
- Conservation: dE/E (model and truth, same fp64 evaluator), |dP|, dL/L about COM, bound fraction (K_i + Phi_i < 0), virial ratio 2K/|U|, fraction outside grid, escapers, Lagrangian radii r10/r50/r90 with truth-replica bands.
- Cost: ms/tick (CUDA-synchronised, median), ns/body/tick, peak GB.

### 1.4 Numbers on record (all H200 NVL, Polaris)
| quantity | value | source |
|---|---|---|
| harness two_body / expB / bh300 err@5/10/20 | .00105/.0031/.0091; .0030/.0081/.0240; .0080/.0209/.0622 | `scatter-baseline.md` |
| dE/E at tick 100 (truth) | .129 (6e-5) / .0108 (1.1e-4) / .0478 (2.5e-4) | same |
| dL/L at tick 100 (truth) | .67 / .40 / .30 (1e-12 .. 2e-4) | same |
| 10^4-tick conservation, N 2-1000 | dE/scale .6-3.0 vs truth 1e-6..4e-3; model hot (virial 4-870 vs truth .7-.85); |dP| 1e-6..3e-2 | `scatter-conservation.md` |
| mesh-only cost, N 1e6-1e8 | 1.6-2.9 ns/body/tick, ~140 B/body (14 GB at 1e8) | `scatter-bh-timing.md` |
| 1e9 BH mesh-only, chunked fp16 body state | 1.33 ns/body/tick (1325 ms/tick), 16.1 GB, 100 ticks, no accuracy | `scatter-bh-1b.md` |
| 1e9 disc galaxy, fp32 state | 1.07 ns/body/tick, 26.1 GB, 1000 ticks, ~3 GPU-h per full set; g2 PASS (dE -.019, bound .979), g1 FAIL (dE -.173, bound .817; truth +.0002/1.000) | experiment log 2026-10-21 |
| launch-bound floor | 2.4-2.6 ms/tick for N <= 3e3 | scaling doc |
| pair term at 1e5 | warm cell 94.5 ms, cold 452 ms; stock chunked 491 ms (O(N^2)) | scaling doc |
| N sweep ratio err@20 / const-vel | cold .02 (N<=316), .39 (3e3), 1.7 (1e4); warm .49 at 1e5 | scaling doc |

### 1.5 Core requirements and what each comparison can honestly exercise
1 generalization (any particle sim), 2 tileable, 3 O(N), 4 interpretability. Only 2 and 3 are quantitatively testable against other codes; 1 only across force laws (done internally: 6 of 7 for the dual tree), and 4 has never been tested. Flagged in section 6.

## 2. Research: established codes and baselines

Cluster note: Polaris is a Slurm H200 cluster; `torch.compile` is unavailable there (no Python.h), so building C/C++/CUDA codes may need a user-space toolchain (conda-forge compilers, HDF5/FFTW/GSL/MPI modules). Not checked in this session; first-week item.

| code / class | what it is | GPU? | fits Augur's planar softened problem? | source (fetched 2026-10-22) |
|---|---|---|---|---|
| Own fp64 direct sum | all-pairs, torch, chunked | yes (exists) | native; reference already used everywhere | repo |
| Own PM (particle-mesh) with the same grid, plain Green's function (kernel-only, no learned weights) | CIC deposit + FFT + gradient | yes (exists as the "kernel-only PM" of the solar run) | native; the ablation "does learning add anything" | repo |
| Barnes-Hut treecode | O(N log N) monopole/quadrupole tree, opening angle theta | torch version unoptimised (`dual_tree.py`) | easy (own softened kernel) | repo |
| REBOUND (WHFast, IAS15, tree, direct modes) | CPU integrator library, `pip install rebound`; softening parameter exists in direct/tree gravity; WHFast symplectic, IAS15 adaptive high-order | no | via z = 0 embedding; independent check of our truth, N <= ~1e4 | [pypi.org/project/rebound](https://pypi.org/project/rebound), IAS15 paper Rein & Spiegel 2015 (via search); WHFast/IAS15 doc details UNVERIFIED |
| GADGET-4 (Springel et al. 2021) | TreePM or momentum-conserving FMM (fast multipole method) gravity, public release, improved force accuracy; force-error tests on isolated Hernquist halo vs direct sum, 90th percentile vs opening angle | no (CPU/MPI; GPU support not mentioned in abstract) | 3D periodic/isolated; softening kernel is a spline (Plummer-equivalent mapping UNVERIFIED); needs HDF5/GSL/FFTW | [arxiv.org/abs/2010.03567](https://arxiv.org/abs/2010.03567) |
| PKDGRAV3 | FMM, 5th-order multipoles, GPU force kernels, 2 trillion particles on Piz Daint | yes (CUDA) | cosmological, periodic, expansion; heavy to adapt | [arxiv.org/abs/1609.08621](https://arxiv.org/pdf/1609.08621) |
| Bonsai | all-GPU Barnes-Hut tree, ~2.8 M particles/s on a GTX480 at theta = 0.75 (2012 hardware); 51-billion-particle Milky Way run on Piz Daint/Titan | yes (CUDA) | open-source status and build on H200/CUDA 12 UNVERIFIED | [arxiv.org/abs/1204.2280](https://arxiv.org/pdf/1204.2280), [1412.0659](https://arxiv.org/abs/1412.0659) |
| FDPS (Framework for Developing Particle Simulators) + Accelerated FDPS | C++ header library, user-defined interaction, N-body example, GPU path; ~27% of Volta peak in the N-body benchmark | yes | good fit: custom kernel (planar, Plummer) is easy | [arxiv.org/abs/1907.02290](https://arxiv.org/pdf/1907.02290), github.com/FDPS/FDPS (not opened) |
| exaFMM (Barba group) / exafmm-t | FMM; original GPU CUDA code from SC11; exafmm-t is CPU kernel-independent (Laplace 3D), last push >4 years before the search snapshot | original yes, exafmm-t UNVERIFIED | Laplace 3D kernel, no Plummer softening out of the box | [bu.edu/exafmm](https://www.bu.edu/exafmm), [joss.theoj.org 10.21105/joss.03145](https://joss.theoj.org/papers/10.21105/joss.03145.pdf) |
| SWIFT | FMM gravity + optional PM for periodic runs, SPH and galaxy formation, consistency and accuracy tests | no | fixed softening, 3D; overkill | [arxiv.org/abs/2305.13380](https://arxiv.org/pdf/2305.13380v1) |
| Abacus | static-multipole-mesh far field + GPU near-field direct sum, median fractional force error ~1e-5, 70 M particle updates/s/node on Summit | yes | cosmological periodic; reference-quality accuracy for cosmology | [arxiv.org/abs/2110.11392](https://arxiv.org/pdf/2110.11392.pdf), accuracy of P(k): [2109.04397](https://arxiv.org/pdf/2109.04397) |
| NBODY6++GPU | collisional direct summation (4th-order Hermite, regularisation), 400-2000x faster than NBODY6 on 320 cores + 32 K20X | yes | only for collisional (star-cluster) tests; licence UNVERIFIED | [arxiv.org/abs/1504.03687](https://arxiv.org/pdf/1504.03687) |
| GIZMO / AREPO | hydro + gravity (tree), FIRE on GIZMO | no | gravity-only comparison is not their use; Prof. Wheeler's FIRE-2/3 galaxies are on GIZMO | [arxiv.org/abs/1409.7395](https://arxiv.org/pdf/1409.7395) |
| GreeM | TreePM (Japanese cosmology code) | not searched | UNVERIFIED, skip |  |
| pmwd, JaxPM, FlowPM (differentiable PM) | JAX/TensorFlow PM libraries with adjoint; FlowPM ~10x faster than a Python PM code on large runs | yes | cosmological 3D periodic; useful as a "plain PM" cross-check of our PM | [arxiv.org/abs/2211.09815](https://www.arxiv.org/abs/2211.09815), [2010.11847](https://arxiv.org/pdf/2010.11847) |

### 2.1 ML surrogates (the other "learned N-body" camp)
Two families, and only one is comparable:
- Step-wise learned simulators (comparable to Augur: same task, state at t -> state at t+dt):
  - GNS (Graph Network-based Simulators; Sanchez-Gonzalez et al. 2020, ICML): message passing on particle graphs, noise injection for long rollouts, generalises to ~10x more particles; fluids/solids, no gravity benchmark in the paper. [arxiv.org/abs/2002.09405](https://arxiv.org/abs/2002.09405)
  - Martinkus, Lucchi, Perraudin 2021 (AAAI), "Scalable Graph Networks for Particle Simulations": hierarchical FMM-inspired graph with O(N) edges; data from a softened (Plummer) symplectic leapfrog simulator; reports phase-space accuracy and energy conservation vs N. Plummer-sphere and direct-summation details UNVERIFIED (abstract/snippet only). [arxiv.org/abs/2010.06948](https://arxiv.org/pdf/2010.06948)
  - Breen et al. 2020 (MNRAS), 3-body deep network, planar equal-mass zero-velocity start, 10,000 runs, up to 1e8 x faster on a bounded interval, energy error 1e-5 after tweaks; compared to Brutus. Fixed-horizon, not a rollout simulator. [arxiv.org/abs/1910.07291](https://arxiv.org/abs/1910.07291)
  - Hybrid PM + learned correction: Lanzieri, Lanusse, Starck 2022 "Hybrid Physical-Neural ODEs for Fast N-body Simulations" (learned isotropic Fourier filter on the PM force, trained through the ODE solver) [arxiv.org/abs/2207.05509](https://arxiv.org/pdf/2207.05509); Payot et al. 2023 "Learning an effective evolution equation for particle-mesh simulations across cosmologies" [arxiv.org/abs/2311.18017](https://ar5iv.labs.arxiv.org/html/2311.18017). These are the closest relatives of Augur's "mesh + learned kernel" and are cosmological 3D.
- Initial-condition-to-final-snapshot emulators (NOT comparable as a simulator): D3M (He et al. 2019, Zel'dovich displacement -> FastPM z = 0) [ar5iv 1811.06533](https://ar5iv.arxiv.org/html/1811.06533); Alves de Oliveira et al. 2020 (tree-N-body target, percent-level P(k) to k ~ 1 h/Mpc); Jamieson et al. 2022 (Field-Level Neural Network Emulator, cosmology-conditioned); NECOLA (Kaushal et al. 2022, COLA -> N-body, ~1% P(k) to k = 1 h/Mpc) [2111.02441](https://ar5iv.labs.arxiv.org/html/2111.02441). They are scored by P(k), halo mass function and cross-correlation, in 3D periodic boxes. We cannot enter that benchmark (section 6).

## 3. Research: benchmark problems with published reference answers

| benchmark | what it gives | regime | usable by Augur? |
|---|---|---|---|
| Hénon / Heggie-Mathieu standard N-body units | G = M = 1, E = -1/4, virial radius 1, crossing time 2 sqrt 2, rms speed 1/sqrt 2 for equal-mass Plummer | any | yes, as a unit conversion for reporting (Augur: M = N unit masses, eps .5; rescale lengths so R_v matches) [en.wikipedia.org/wiki/N-body_units](https://en.wikipedia.org/wiki/N-body_units), Heggie lecture notes (via search) |
| Plummer sphere, Hernquist, King models | equilibrium ICs; stationarity test; force-error test vs direct sum (GADGET-4 uses a 1M-particle isolated Hernquist halo, 90th-percentile error vs opening angle; Dehnen 2002 uses mean and 99th percentile) | collisionless | yes in planar form: projected Plummer already exists (globular); planar Hernquist/King to be written. Dehnen 2002 [astro-ph/0202512](https://arxiv.org/pdf/astro-ph/0202512), GADGET-4 above |
| Burrau / Pythagorean 3-body (masses 3,4,5 at rest on a 3-4-5 triangle) | Szebehely & Peters 1967 reference through tau ~ 67; ends as a binary plus escaper; chaotic with close approaches | collisional | planar already; softened version differs from the unsoftened classic, so reference = our truth validated against REBOUND, not the 1967 numbers. Mass ratio 5:3 inside the 3:1 limit. [ucolick Burrau page](https://www.ucolick.org/~laugh/oxide/projects/burrau.html), [arxiv.org/abs/2001.01221](https://arxiv.org/pdf/2001.01221) |
| Cold collapse (uniform or Plummer sphere with low virial ratio) | classic violent-relaxation test; sensitivity to softening and integrator | collisionless-ish | yes: `scatter_scaling` cold family is this (vfac .3) in 2D; no published numerical table found in this session (UNVERIFIED) |
| Kepler 2-body, binaries | period / precession / dE | any | exists (binary) |
| Disc stability (Fall-Efstathiou 1980, Sellwood-style) | bar formation timing, disc heating; Sellwood's papers compare grid, tree, basis-function codes and conclude grid/basis codes are efficient for disc stability and that quiet starts help | collisionless | `galaxy_ic.py` is a planar disc + bulge, no dark matter; direct tie to Sellwood's methodology; specific Fall-Efstathiou numbers NOT retrieved (UNVERIFIED). [astro-ph/9612087](https://arxiv.org/pdf/astro-ph/9612087), [astro-ph/0601277](https://arxiv.org/pdf/astro-ph/0601277) |
| AGORA isolated disc (Kim et al. 2016) | 9 codes on one Milky-Way-mass disc IC, common analysis (yt), public snapshots at 0 and 500 Myr | gravity + hydro + feedback, 3D | no: gas, DM halo, 3D. Public data release [arxiv.org/abs/2001.04354](https://arxiv.org/pdf/2001.04354) |
| Santa Barbara cluster | 12 codes, one CDM cluster, gas and DM profiles; DM agrees best | cosmological | no (3D cosmological, hydro); original data reported not public (one group's statement, UNVERIFIED). [astro-ph/9906160](https://ar5iv.labs.arxiv.org/html/astro-ph/9906160) |
| Aquarius (Springel 2008), Via Lactea | 6 galactic haloes, up to ~3e5 subhaloes | cosmological | no; public data availability not confirmed here |
| Quijote (Villaescusa-Navarro 2020) | 44,100 N-body sims, >7,000 cosmologies, ~1 PB, snapshots, halo catalogues, summary statistics | cosmological periodic | no (3D periodic expansion). [arxiv.org/abs/1909.05273](https://arxiv.org/abs/1909.05273v1), docs at quijote-simulations.readthedocs.io |
| Zel'dovich pancake | analytic 1D solution rho = rho0 [1 - ((1+z_c)/(1+z)) cos(kx)]^-1 until caustic; PM codes report L1/L2/Linf error and convergence order | cosmological, 1D | no (needs expansion, periodic, 1D); PM unit test only. [astro-ph/0403233](https://arxiv.org/pdf/astro-ph/0403233) |
| Code comparisons for P(k) | Schneider et al. 2016: Ramses, PKDGRAV3, Gadget3 agree to 1% at k ~ 1 h/Mpc, 3% at 10; Euclid 2023 massive-neutrino comparison, 11 implementations sub-percent; Heitmann 2005: 5% best case, >10% worst | cosmological | model for how a comparison table is written, not for us |
| Convergence criteria (Power et al. 2003) | trustworthy radius needs timestep << orbital time, acceleration < softening scale, enough enclosed particles that two-body relaxation time exceeds the run (~3000 particles inside r for 10^6 density contrast) | collisionless halo | the relaxation criterion applies to our N <= 1e3 clusters. [astro-ph/0201544](https://arxiv.org/pdf/astro-ph/0201544) |
| Optimal softening (Dehnen 2001) | Plummer softening has larger force error than compact-kernel softening; optimal eps balances shot noise vs bias | collisionless | explains why a Plummer-softened comparison favours codes with kernel softening in P(k)-style metrics. [astro-ph/0011568](https://arxiv.org/abs/astro-ph/0011568) |
| Reviews | Dehnen & Read 2011: collisional N > 1e6, collisionless > 1e9 particles; covers algorithms, softening, relaxation, timesteps | both | primary reading for the paper's related-work section. [arxiv.org/abs/1105.1082](https://arxiv.org/pdf/1105.1082) |

## 4. Standard metrics and fair-comparison pitfalls

### 4.1 Accuracy metrics used in published comparisons
- Force accuracy vs direct summation: relative force error |a - a_ref| / |a_ref| per particle, reported as median, 90th, 99th percentile (or mean) vs the algorithm's accuracy knob (opening angle theta, multipole order p, mesh size) [GADGET-4, Dehnen 2002]. Cost on the x-axis gives the standard Pareto plot.
- Conservation: |dE/E|, |dL/L|, |dP|, COM drift, for isolated systems; virial ratio 2K/|U| settling to 1; bound fraction.
- Structural: Lagrangian radii vs time, density profile, velocity dispersion/anisotropy profile, core radius, escaper count; for discs: bar strength (A2 Fourier amplitude), pattern speed, Toomre Q(t), rotation curve, radial profile; for cosmology: P(k) ratio, bispectrum, halo mass function, two-point correlation.
- Resolution/convergence: the Power et al. radius.
- Cost: time-to-solution at fixed accuracy, particle updates per second per GPU (Abacus: 70 M/s/node on Summit; Bonsai 2012: 2.8 M/s on a GTX480), memory per particle (Augur ~140 B/body at 1e8 stock, 16-26 GB at 1e9 chunked), strong/weak scaling.

### 4.2 Pitfalls that apply to Augur specifically
1. Softening: Augur truth is Plummer with eps .5. GADGET-family codes use spline softening, so a "same eps" run is a different force law at r < ~2.8 eps (UNVERIFIED mapping; check the code's softening options before use). Compare against a code only after setting its kernel to Plummer, or build the baseline ourselves (own torch tree/FMM/PM all take the same kernel). Dehnen 2001: Plummer is the worst softening for force bias, so an unmatched comparison unfairly favours the other code.
2. Integrator: Augur has 1 force eval per tick (dt .1); its "truth" uses 4 substeps. The exact force with Augur's own 1-eval integrator ("t1") already costs dE/E .016 (bh300, tick 100) to .8 (tick 1e4). Baselines must be run at the SAME dt and 1 eval per tick as well as converged, or the integrator is blamed on the learned force (or vice versa).
3. Grid: h = .5 = eps. A plain PM at h = .5 cannot resolve forces below ~1 cell; Augur's pair term (pp 2.0, knn 16) is what fixes the near field, and it is removed at >= 1e6 (mesh-only runs). At 1e9 the model is a PM code with a learned kernel, so the right competitor is another mesh code, not a tree.
4. Chaos: per-body positions diverge past ~20-50 ticks even between fp32 and fp64 (floors in `chaos_floors.json`); beyond the horizon only statistics (r50, E, bound fraction, profiles) are comparable, with truth-replica bands, not error vs one trajectory.
5. Collisional vs collisionless: relaxation time ~ N/ln(gamma N) crossing times (textbook scaling, not re-verified here). At N = 300 with eps .5 the system relaxes within the 1000-tick runs; direct summation sees that, a mesh does not. Collisionless tests (smooth ICs, large N) are the fair arena for a mesh model; collisional tests (Pythagorean 3-body, binary hardening, core collapse) favour regularised direct summation (NBODY6++GPU, REBOUND IAS15) by construction.
6. Hardware/units: literature particle-update rates are on 2012-2019 GPUs or CPU nodes; only same-machine runs are claims. Wall time of our existing torch tree/dual-tree is unoptimised and must not be used as a speed baseline.
7. Dimensionality: planar force-in-3D-law vs 3D codes: results for 2D mass distributions (surface density) differ from 3D spheres at the same N; do not quote Hernquist/Plummer sphere numbers from papers as if they applied.
8. Selection: choose checkpoint on held-out seeds (standing rule), and report E-vs-E noise (1.01) when comparing runs.

## 5. Options, ranked by (value for credibility) / (effort)

Common conventions: seed 4738 (dev), 9100/9200/9300 held out; all Polaris jobs follow the one-GPU-at-a-time and batch-the-arms rules; every experiment gets an experiment-log entry (failures included); fresh unbiased-subagent visual review on any rollout output.

### Option 1. Reference validation: our fp64 truth vs independent codes (credibility prerequisite)
- Core requirements: none directly (it validates the instrument).
- Scenes: Kepler 2-body (e = .5, 100 orbits), Pythagorean 3-body (softened, scaled so min separation >> eps for the first 20 time units, then also with close approaches), planar Plummer N = 1000 (globular generator) for 100 time units, cold Gaussian N = 300.
- Baselines: REBOUND (direct gravity with softening, IAS15 and WHFast, and tree mode with theta sweep) on CPU; optionally GADGET-4 FMM on a 3D z = 0 embedding if its softening can be forced to Plummer.
- Metrics: position agreement over time (should sit at the reference-noise floor), dE/E, dL/L, tree-mode force percentiles vs our direct sum, our t1/t4/t16 substep sensitivity (already in `scatter-conservation.md`).
- Expected: agreement to numerical noise for N <= 1e3 until chaos; tree mode differs at the theta-set accuracy.
- Pass: our t4 truth within 1e-6 relative of IAS15 for Kepler and N = 3 to t = 20; Plummer r10/r50/r90 within the REBOUND-IAS15 replica spread. Kill/stop: any disagreement above the reference-noise floor, then the truth, not Augur, is the first fix (this would invalidate prior err@k).
- Cost: 0 GPU-h (CPU, N <= 1e3), 1-2 days engineer effort.
- Risk: softened REBOUND semantics (Plummer, per-pair) must be confirmed in docs (UNVERIFIED); WHFast is a symplectic planetary integrator (Wisdom-Holman), not suited for the cluster scenes, use IAS15/leapfrog there.

### Option 2. Force-error vs cost Pareto on standard static snapshots (the GADGET-4 / Dehnen-style plot)
- Requirements: O(N) (cost vs N), tileable (single grid vs 8x8 tiles), generalization (several density profiles without retraining).
- Scenes (one force evaluation each, no rollout so no chaos): Gaussian cluster sigma 8 (existing), planar Plummer a = 2, 4, 8 (a >= 2 is the stated validity), planar Hernquist/King (small new IC function, ~50 lines), disc g2 snapshot at tick 0/300, N = 1e3, 1e4, 1e5, 1e6 (exact reference on a random 1e4-target subset against all sources, fp64; used already at 5e4 x 1e6 in the galaxy runs).
- Candidates on the same snapshots: Augur stock Exp E (with and without pair term), Augur + R1b tiled, plain PM at same grid (kernel-only), Barnes-Hut/dual-tree (theta sweep 0.3-1.0, monopole, quadrupole; own torch), FMM (own or FDPS GPU, if built), REBOUND tree (CPU, <= 1e5).
- Metrics: relative force error median / 90th / 99th percentile (bodies; separately core vs halo), vs wall time per evaluation and peak GB, same H200; momentum residual sum m a.
- Expected (from existing data): tick-1 force error .05-.14 acceleration-weighted at N up to 2e4 (cosine .99+), i.e. 5-14% against a Barnes-Hut at theta .5-.7 that typically reaches 1e-3..1e-2 (typical literature range, UNVERIFIED for our kernel). Plain PM at h = .5 has larger near-core error; Augur's learned kernel+U-Net should beat it. At N >= 1e6 only mesh-only is possible: compare vs PM only.
- Pass: Augur (with pair term) beats the same-grid plain PM in median and 90th-percentile force error by >= 2x (value of learning) at N = 1e3-1e5 on all three held-out profile families; and sits on the Pareto front (not dominated in error and time simultaneously) against Barnes-Hut at some N in [1e5, 1e6]. Kill: plain PM at the same grid equals Augur within 20% at the 90th percentile (the learned part adds nothing measurable on force; then the paper claim must shift to speed and tiling). Honest prior: fail the "beat the tree on accuracy" half; accuracy will be 5-100x worse than a tree at equal wall time ONLY IF the tree is well optimised, so the wall-time axis needs an optimised GPU tree (Option 4 dependency).
- Cost: 3-5 GPU-h (exact reference at 1e6 targets dominates, subset), 3-4 days.
- Risks: far-kernel breakdown beyond r > 64 (use R1b branch, state which model); mass/profile OOD for the Hernquist cusp (a < 2 is 107x worse in the zero-shot table, a dense-core fail is expected and must be shown, not hidden).

### Option 3. Collisionless rollout report card against exact, tree and PM
- Requirements: generalization (scene types), tileable (tile vs single on the same scene), O(N) via mesh-only at large N.
- Scenes: (a) planar Plummer equilibrium N = 1e3-1e4 at 5 and 20 crossing times (must stay stationary), (b) cold collapse (existing vfac .3 family, N 1e3, 1e4), (c) disc galaxy g2 (existing IC, N = 1e5; truth at 1000 ticks already run) with bar/spiral metrics, (d) 8x8-tiled version of (a) or (c) at N = 1e5 vs single grid (reuse the seam-video setup).
- Baselines: exact fp64 (truth), plain same-grid PM, Barnes-Hut leapfrog at the same dt and one eval per tick (own torch; theta .5), REBOUND-tree (CPU) at N <= 1e4 as an independent code, GADGET-4/FMM optional.
- Metrics: err@5/10/20 vs truth with chaos floors; r10/r50/r90 vs time inside truth-replica bands; E(t), virial ratio, bound fraction; |dP|, dL/L; A2 bar amplitude and its onset tick, Toomre Q(t), rotation curve at tick 0 and 1000 (g2); fraction outside grid.
- Expected (existing numbers): err@20 ratio .02-.05 for N <= 1e3; globular dE/E 3-18% at 1000 ticks (truth < .4%); g2 at N = 1e5: dE -.04, bound .93-.98, tracks through tick ~300 (g1 fails). Plain PM at h = .5 will probably heat less or more in the core (unknown, this is the experiment). Dehnen & Read note collisionless regimes are the accepted use of PM-class methods; conservation parity with leapfrog/tree codes will NOT be reached (tangential force 3-40%, curl .13).
- Pass (pre-registered): over the first 100 ticks (the trained horizon, 10 time units) Augur's r50 and E inside truth-replica band >= 90% of ticks on Plummer N = 1e3 and cold collapse, and err@20 <= 1.5x the same-grid plain PM's; over 1000 ticks g2 at 1e5 keeps bound >= .90 and |dE| <= .10 (same bar as the 1e9 run) and bar onset within 20% of truth tick. Kill: Augur worse than same-grid PM on r50 and E at tick 100 on both Plummer and collapse; or tiled (d) differs from single by more than the 1.1x err@20 bar already used.
- Cost: ~10-15 GPU-h (exact truth of 1000 ticks at 1e5 was ~90 min; tree/PM arms are minutes), 1 week.
- Risk: long-horizon conservation is the model's documented weakness (dE/E O(1) by 1e4 ticks, virial 4-870); expect partial fail past tick 300 and report it as such. Equilibrium-IC generation in 2D with 3D-law softening: Jeans dispersion needs checking (the galaxy code derived v_c from enclosed mass and found ~10% underestimate beyond R = 6); an IC that is not in equilibrium for truth invalidates the stationarity test, so first run truth for stationarity.

### Option 4. Throughput and scaling head-to-head at 1e5-1e9 on one H200
- Requirements: O(N), tileable.
- Scenes: uniform + clumpy + disc g2 at fixed density, N = 1e5, 1e6, 1e7, 1e8, 1e9 (mesh-only, chunked, as in the 1e9 runs; fp16 caveat stated).
- Baselines: own well-optimised GPU PM at the same grid (CIC + cuFFT, torch), at least one optimised GPU tree/FMM: Bonsai (build UNVERIFIED), FDPS GPU N-body sample (UNVERIFIED build), or PKDGRAV3/Abacus on a periodic uniform box as an upper-end reference (both cosmological; use only for throughput, state it). Published rates (Abacus 70 M updates/s/node on Summit, Bonsai 2.8 M/s on a 2012 GPU) quoted as context only.
- Metrics: ns/body/tick (median, CUDA-synchronised), peak GB, bytes/body, slope in log-log (O(N) check: 0.96-0.99 on record), time-to-solution for 1000 ticks, plus force-error at the matched cost from Option 2.
- Expected: Augur mesh-only 1.07-1.33 ns/body/tick at 1e9 (26 / 16 GB); a tuned PM on the same H200 will be the same order (mesh scatter/gather bound); an optimised GPU tree at ~10-100 ns/body (assumption from Bonsai's published rate scaled by GPU generation, UNVERIFIED). The honest result is "same order as plain PM, 1-2 orders faster than trees, with accuracy per Option 2".
- Pass: Augur within 1.5x of the plain same-grid PM cost at every N (learned layers do not dominate) and >= 5x faster than the best tested tree at equal N >= 1e6; Kill: Augur > 3x slower than plain PM at 1e8 (U-Net cost dominating), or any baseline faster at equal-or-better force error (then the speed claim is dropped).
- Cost: ~8-12 GPU-h (the 1e9 cells are 5-25 min each), 5-7 days (mainly building a GPU tree baseline).
- Risks: the 1e9 mesh-only run has no accuracy, so the cost claim stays separate from the accuracy claim (resume-line rule); building third-party GPU codes on Polaris may fail (compiler/toolchain); apples-to-oranges in precision (fp16 state vs fp32/fp64).

### Option 5. ML-baseline head-to-head on a shared planar gravity task
- Requirements: generalization (same data, several architectures), O(N), interpretability (only qualitative).
- Scenes: training data from `gravity_sim.make_dataset` (N 10-300, scale-init, cold Gaussian, existing), test on expB, bh300, globular N = 300/1000, plus N-extrapolation to 1e4.
- Baselines: GNS (DeepMind code or PyG port), Martinkus hierarchical graph net (code availability UNVERIFIED), the repo's own token model (known to fail at 100-1000 bodies) and CentralForceDynamics (err@20 .0054 at 100-1000 bodies), and a hybrid PM + learned Fourier filter (Lanzieri et al.) reimplemented in torch.
- Metrics: err@5/10/20 with floors, dE/E, |dP| at 100 and 1000 ticks, cost vs N, parameter count, training GPU-h.
- Expected: Augur's err@5/10/20 .003/.008/.024 on expB is better than anything unmodified GNS likely reaches at equal training time (UNVERIFIED expectation; no published GNS gravity result found), but CentralForceDynamics is conservative-by-construction and wins on energy. Honest message: "mesh + learned kernel is the competitive learned approach; central-force priors win conservation".
- Pass: Augur err@20 <= 0.5x best of GNS/Martinkus/hybrid at matched training GPU-h on expB and bh300; Kill: any baseline matches within 20% at lower cost.
- Cost: 30-60 GPU-h (training four models, 50 epochs each by default), 1.5-2 weeks. Highest cost, lowest cheapness: academic audience, but comparisons against weak reimplementations are not credible (state hyperparameter effort).
- Risk: baseline under-tuning accusation; no standard benchmark exists.

### Ranking (value for credibility per effort)
| rank | option | value | effort | GPU-h | first claim it supports |
|---|---|---|---|---|---|
| 1 | Reference validation (REBOUND, optional GADGET-4) | high (prerequisite) | 1-2 d | 0 | "our truth is an independent-code-verified N-body" |
| 2 | Force-error/cost Pareto on static snapshots | high | 3-4 d | 3-5 | "accuracy per cost vs tree and PM at fixed N" |
| 3 | Collisionless rollout report card | high | ~1 wk | 10-15 | "behaves like a collisionless code for ~100 ticks; tiled = single" |
| 4 | Throughput/scaling head-to-head to 1e9 | medium-high | 5-7 d | 8-12 | "same cost class as PM, faster than trees" |
| 5 | ML-baseline head-to-head | medium | 1.5-2 wk | 30-60 | "best of learned approaches on this task" |

Not recommended now: cosmological box (P(k), halo mass function, Quijote/CAMELS/Illustris, Zel'dovich), Santa Barbara, Aquarius, AGORA, Heggie collisional star-cluster tests (core collapse, binary hardening).

## 6. Claims we cannot honestly make (today)
- Accuracy at 1e8-1e9 bodies: no exact reference; 1e9 runs are mesh-only timing/stability, bound fraction and dE vs exact only at 1e5 (g1 FAILs at every N).
- 3D or cosmological results (periodic, expanding, P(k), halo mass function): the scatter model is planar, non-periodic, fixed eps/h; the 3D central-force model is a different architecture (conservative by construction).
- Energy or angular-momentum conservation comparable to leapfrog/tree/FMM codes: model dE/E .01-.13 at tick 100 vs truth 1e-4; dL/L .3-.7 vs 1e-12..2e-4; unbounded growth by 1e4 ticks; momentum only is exact (momfix).
- Long-horizon trajectories: trained horizon ~20 ticks, per-body error ceiling ~50 ticks (chaos), statistics only to ~300 ticks (g2) before edge leakage.
- Collisional dynamics (hard binaries, close encounters, core collapse): not tested; softened eps .5 hides them; Pythagorean 3-body is only meaningful in the softened, scaled form.
- Different eps, cell size, mass ratio > 3:1, dominant body > 3x, dense cores (sigma <= 2, Plummer a <= .25): outside the validity table without a per-interaction retrain.
- "Learns any simulator" and "interpretability": the kernel-only PM ablation (Option 2) is the first quantitative evidence of what learning adds; the interpretability requirement is untested.
- Speed vs other codes from the literature: those rates are on other hardware; the existing `dual_tree` wall times are unoptimised and "est. 3-10x slower than tuned analytic GPU codes" in analysis-process.md is a recollection, not a measurement.
- Learned vs other ML emulators: none of D3M/Jamieson/NECOLA solve the same task (IC -> snapshot, cosmological), so no head-to-head is possible there.

## 7. Minimum baseline install set and first-week plan

Minimum set (in order):
1. Own fp64 direct sum (exists) and own same-grid kernel-only PM (exists, needs a cleaned entry point).
2. REBOUND (`pip install rebound` in a venv on Polaris login or local; CPU).
3. Own Barnes-Hut with Plummer kernel on GPU written for speed (quadrupole, theta sweep); the existing `dual_tree.py` is a starting point but unoptimised.
4. One third-party optimised code, chosen after a 1-day build trial in this order: FDPS (C++ header library, GPU path, user kernel, easiest to make Plummer), GADGET-4 (CPU, FMM mode, force-error tests documented), Bonsai (CUDA, UNVERIFIED build). Only build one; keep the others as citations.
Not installed in v1: PKDGRAV3, Abacus, SWIFT, NBODY6++GPU, GIZMO/AREPO (cosmological or collisional or hydro; no matching test).

First week (engineer ~5 days, GPU ~6-10 h, one-GPU-at-a-time):
- Day 1: decisions (below); register pass/kill for Options 1-3 in the experiment log before any run; check log "already tried" list. Install REBOUND; write `scripts/baselines/` entry points for direct, PM, BH with a common `Snapshot -> acceleration` interface. Confirm REBOUND softening semantics.
- Day 2: Option 1 runs (Kepler, Pythagorean, Plummer, cold collapse) on CPU; write up.
- Day 3: planar Plummer/Hernquist/King IC functions; snapshot set (1e3-1e6) and exact fp64 subset reference; PM and BH arms.
- Day 4: Polaris job for Option 2 (single batched job: 3 profiles x 4 N x models, ~2-3 GPU-h); check GPU utilisation; summarise percentiles and Pareto.
- Day 5: toolchain trial for FDPS or GADGET-4 build on Polaris; write Option 2 results + Option 3 pre-registration; fresh unbiased-subagent read of any rendered rollout.

## 8. Decisions needed from the user
1. Scope of the headline claim: planar, collisionless, ~100-tick horizon, mesh-only at >= 1e6 (what exists), or invest in a 3D / cosmological model first (months, not weeks)?
2. Which Augur configuration is the one compared: stock Exp E, or tiled + R1b far kernel with per-interaction retrain (needs the retrain run first)?
3. Approve Option 1 as a prerequisite and its pass/kill (truth within 1e-6 of IAS15 for N <= 3 to t = 20)?
4. Approve the "same-grid plain PM" ablation as the primary baseline and kill rule (Option 2: Augur must beat it 2x at the 90th percentile), accepting the paper claim shifts to speed + tiling if it does not?
5. Collisional tests (Pythagorean beyond t = 20, binary hardening, core collapse) in or out?
6. Build effort for an optimised third-party GPU baseline (FDPS / GADGET-4 / Bonsai): approve one day to trial the build on Polaris?
7. ML baselines (Option 5): in the paper or later, and is a 30-60 GPU-h budget acceptable?
8. Is there an external audience that decides baselines (e.g. Prof. Wheeler's group uses GIZMO; a GIZMO gravity-only comparison needs her input), or is the target a methods paper where Options 1-3 suffice?
9. Unit convention for reporting: keep sim units (G = 1, eps .5, dt .1) or convert to Hénon units with N-scaled mass?
10. The accuracy-vs-cost wall-time axis uses a baseline we build: acceptable, or must it be a third-party code?

## 9. Sources
- Dehnen & Read 2011, N-body simulations of gravitational dynamics: https://arxiv.org/pdf/1105.1082
- Springel et al. 2021, GADGET-4: https://arxiv.org/abs/2010.03567
- Potter, Stadel, Teyssier, PKDGRAV3: https://arxiv.org/pdf/1609.08621
- Bedorf et al., Bonsai: https://arxiv.org/pdf/1204.2280 ; Milky Way 51 billion: https://arxiv.org/abs/1412.0659
- FDPS accelerated: https://arxiv.org/pdf/1907.02290
- REBOUND: https://pypi.org/project/rebound
- exaFMM: https://www.bu.edu/exafmm ; exafmm-t JOSS: https://joss.theoj.org/papers/10.21105/joss.03145.pdf
- SWIFT: https://arxiv.org/pdf/2305.13380v1
- Abacus: https://arxiv.org/pdf/2110.11392.pdf ; power-spectrum accuracy https://arxiv.org/pdf/2109.04397
- NBODY6++GPU: https://arxiv.org/pdf/1504.03687
- pmwd: https://www.arxiv.org/abs/2211.09815 ; FlowPM https://arxiv.org/pdf/2010.11847
- GNS: https://arxiv.org/abs/2002.09405 ; Martinkus et al.: https://arxiv.org/pdf/2010.06948 ; Breen et al.: https://arxiv.org/abs/1910.07291
- Hybrid physical-neural ODEs: https://arxiv.org/pdf/2207.05509 ; Payot et al.: https://ar5iv.labs.arxiv.org/html/2311.18017
- D3M: https://ar5iv.arxiv.org/html/1811.06533 ; NECOLA: https://ar5iv.labs.arxiv.org/html/2111.02441
- Santa Barbara: https://ar5iv.labs.arxiv.org/html/astro-ph/9906160 ; AGORA II: https://arxiv.org/pdf/1610.03066 ; AGORA data: https://arxiv.org/pdf/2001.04354
- Power et al. 2003: https://arxiv.org/pdf/astro-ph/0201544 ; Dehnen 2001 softening: https://arxiv.org/abs/astro-ph/0011568 ; Dehnen 2002 tree: https://arxiv.org/pdf/astro-ph/0202512
- Quijote: https://arxiv.org/abs/1909.05273v1 ; Aquarius: https://arxiv.org/pdf/0809.0898
- Sellwood: https://arxiv.org/pdf/astro-ph/9612087 , https://arxiv.org/pdf/astro-ph/0601277
- Burrau problem: https://www.ucolick.org/~laugh/oxide/projects/burrau.html ; https://arxiv.org/pdf/2001.01221
- N-body units: https://en.wikipedia.org/wiki/N-body_units
- Zel'dovich pancake test: https://arxiv.org/pdf/astro-ph/0403233
- Code comparisons: Schneider 2016 / Euclid 2023 / Heitmann 2005 via search snippets (Heitmann: https://arxiv.org/pdf/astro-ph/0411795; Euclid: https://arxiv.org/abs/2211.12457v1); Schneider 2016 URL not captured (UNVERIFIED citation details).
