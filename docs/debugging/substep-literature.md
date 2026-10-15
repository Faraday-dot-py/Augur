# Substep / multi-rate integration literature survey (2026-10-14, uncommitted)

Researcher pass, outside literature only. No code, no compute. Premises from long-rollout-root-cause.md (integrator floor: exact force + 1 Verlet/tick vs 4-substep truth gives err@100 .25 expB / 4.0 clusters; 2 exact substeps cut it 5-7x; error carried by nn<1 bodies) and error-correction-plan.md "Rejected" list (nothing below repeats it; parareal is already rejected, so it is not proposed).

Verification tags: FT = full text fetched (WebFetch returns a tool-side summary, not my own reading of the PDF; PDFs were unreadable locally, so HTML/ar5iv was used); AB = abstract or search snippet only; MEM = recalled from memory, not checked this session. Treat AB and MEM as leads.

## Summary (read this)

Classical N-body practice answers the question directly: give each body (or pair) a timestep from a local criterion, put timesteps on power-of-two rungs, apply the slow force once per large step and the fast force on the small steps (hierarchical Hamiltonian splitting; RESPA in MD is the same idea). Cost is proportional to the number of flagged bodies times their rung depth, so it stays O(N) if rung depth is capped. Augur already has the natural split: the g128 mesh force is the slow/expensive part; the pp=2/knn=16 pair term is local, cheap, and re-evaluable per substep.

Findings that matter for Augur:
1. Pairwise symmetric kicks conserve momentum exactly under individual timesteps (Pelupessy+ 2012, FT); the criterion must be time-symmetrised (Hut-Makino-McMillan 1995 AB; Pelupessy explicit form tau/(1 - dtau/dt /2), FT) or energy drifts. Per-particle accel-based criteria with non-symmetric kicks drift (Quinn+ 1997, FT). Augur is not symplectic already, and the gate is dE rows, so the practical rule is: symmetric pair criterion, block rungs, mesh impulse on rung 0 only.
2. The one MTS-for-ML-potential paper (Cattin+ 2025, FT) found the cheap-fast/expensive-slow split works (4x, outer step 4-6 fs) but fails from long-tail force errors in the cheap model, not from resonance. For Augur: the near-field used in substeps must agree with the full model's near-field; mismatch tails will destabilise.
3. Learned residual correctors on a coarse step (hypersolvers, FT) need no extra force evaluations and are trained on dense one-step residuals, which sidesteps the tail-dominated rollout loss that root-cause cause #3 blames. Risk: non-symplectic, dt-specific, and the current learned force may already absorb part of the floor (pair probe: E -20% at r=.1), so the target shifts.
4. Learned/neural close-encounter dynamics are fragile: neural residuals failed to approach analytic results in a 2026 falsification study (AB); Hamiltonian nets hard to train, authors fall back to a numerical solver when the net looks wrong (Horn+ 2023, AB). Do not learn the integrator for close pairs first.
5. Nothing found that routes extra compute per particle inside a learned particle-mesh simulator. Adaptive-compute nets (MoD, ACT/APGCN, MeshGraphNets remeshing) transfer only as a pattern: fixed-capacity top-k selection gives static shapes and a density-independent cost cap.

Ranked top 6 (details below): (1) pair-local RESPA-style substeps with frozen mesh impulse; (2) fixed-budget top-k selector scored by an embedded-pair error estimate; (3) hypersolver residual on the pair step; (4) hybrid high-order sub-integrator for close components; (5) global 2-substep as control and fallback; (6) learned structure-preserving pair map. Cheapest first test needs no training: oracle experiment with exact forces, mesh part frozen, pair part substepped only for flagged bodies, versus global 2-substep and truth.

Not found / unverified: any learned particle simulator that reports substepping effect on rollout error; any measured force-vs-trajectory trade for adaptive substeps in a PM code.

## 1. Classical

| Ref | Finding (one line) | Tag |
|---|---|---|
| Aarseth 1963, MNRAS 126, 223 (individual timesteps) | Origin of individual/block timesteps in N-body | MEM |
| Hut, Makino, McMillan 1995, ApJ 443, L93 "Building a better leapfrog" | Choosing dt time-symmetrically makes variable-step leapfrog/Hermite avoid secular energy error; tested on e~high Kepler | AB |
| Preto & Tremaine 1999, AJ, astro-ph/9906322 | Explicit symplectic leapfrog with dt a function of position only (time transformation dt~1/V); naive step-size control destroys symplecticity | AB (search snippet; ar5iv page empty) |
| Quinn, Katz, Stadel, Lake 1997, astro-ph/9710043 | Per-particle KDK with power-of-two rungs. Criteria tested: eta/sqrt(G rho), eta sqrt(eps/a), eta eps/v. Density-criterion DSKD shows no secular drift; SDKD drifts; velocity criterion not Galilean invariant; acceleration criterion fine for cosmology. Measured speedup 2.7 vs fixed step (max ~13 predicted) | FT |
| Zemp, Stadel, Moore, Carollo 2007, MNRAS 376, 273, astro-ph/0606589 | Timestep from enclosed density, with eccentricity correction C(e) for close pairs; without it the e=.9 test gained ~240x more energy error; standard accel+softening criterion circularises tight orbits | FT |
| Springel 2005 (GADGET-2), astro-ph/0505010; Springel+ 2021 (GADGET-4), 2010.03567 | dt ~ sqrt(2 eta eps/|a|); GADGET-4 adds optional hierarchical Hamiltonian splitting that decouples short-timescale dynamics | AB |
| Pelupessy, Janes, Portegies Zwart 2012, New Astron., 1205.5668 | Hierarchical splitting into pairwise kicks: momentum/angular momentum to machine precision. Criteria tau_ij = eta1 sqrt(r^3/(m_i+m_j)) and eta2 r/v; symmetrised tau/(1 - dtau/dt/2). Energy error ~eta^2. Cannot save work when everything is bound pairs. Symmetrisation approximate, jumps at binary interactions. No explicit O(N) claim | FT |
| Janes, Pelupessy, Portegies Zwart 2014, A&A 570, A20, 1407.7105 | Partition into connected components of a timestep graph; each component integrated independently with its own step | AB |
| Farr & Bertschinger 2007, ApJ, astro-ph/0611416 | Variational (fourth-order) integrator; power-of-two block steps do not worsen symplecticity error; ~1.5 orders better than standard codes at equal force evals | AB |
| Hernandez & Bertschinger 2015, 1503.02728 | Symplectic Kepler-solver integrator uses fixed step because adaptive steps break symplecticity | AB |
| Duncan, Levison, Lee 1998, AJ 116, 2067 (SyMBA) | Wisdom-Holman plus recursive timestep subdivision for bodies in close encounter only; naive variable dt destroys symplecticity | AB |
| Hernandez & Dehnen 2023, MNRAS 522, 2301.06253; 2024, MNRAS 530, 2401.07113 | Reversible integrator switching and multi-timestep WH: each pair gets its own step, reversible if block-synchronised; up to 2.6x faster than SyMBA-based code on close encounters | AB (2401.07113 abstract fetched) |
| Rein & Spiegel 2015, MNRAS 446, 1424, 1409.4779 (IAS15); Rein & Tamayo 2015, 1506.01084 (WHFast with hybrid switch to IAS15 on close encounter) | 15th-order adaptive non-symplectic integrator; hybrid symplectic-to-high-order switching for encounters | AB |
| Wisdom & Holman 1991; Yoshida 1990; Forest & Ruth 1990 | Kepler-drift splitting; composition higher-order symplectic schemes (negative substeps) | MEM |
| Biesiadecki & Skeel 1993; Ma & Izaguirre | Impulse-MTS resonance instability when slow step approaches half the fastest period | MEM (not searched) |
| Dehnen & Read 2011, 1105.1082 (review) | General reference for criteria and cost | AB (title only) |

Timestep criteria in Augur terms: truth uses a fixed 4 substeps (dt=.025) and a softened kernel, so singularity regularisation (Preto-Tremaine, KS) buys nothing. The relevant criteria are free-fall tau_ij = eta sqrt(eps_eff^3 or r^3 / m_ij) and flyby r/v, symmetrised; a rung is the smallest 2^-n dt such that tau_i >= dt/2^n. Aligning rungs to the truth substep grid (1, 1/2, 1/4 of dt) makes flagged bodies reproduce the truth integrator exactly for those bodies.

Pitfalls to keep: (a) criterion asymmetric in i,j gives energy drift; (b) rung changes mid-orbit break reversibility, so change rungs only at synchronisation points; (c) velocity criteria are not Galilean invariant (Quinn); (d) impulse MTS resonates when pair orbital period approaches the outer step (MEM): dt=.1 coarse tick versus close-pair periods may hit this; (e) a body in a tight bound pair puts both partners on the fast rung, so cost is pairs x rung depth.

Cost vs density (reasoned, not measured here): free-fall dt ~ 1/sqrt(rho_local) so substeps per body per tick ~ sqrt(rho) until the softening cap; total cost O(N sqrt(rho_local)) per tick uncapped, O(N * 2^r_max) capped. Not density-blind unless the budget in design 2 is imposed.

## 2. Learned integrators and time stepping

| Ref | Finding | Tag |
|---|---|---|
| Poli, Massaroli, Yamashita, Asama, Park 2020, NeurIPS, 2007.09601 (hypersolvers) | z_{k+1} = z_k + eps psi + eps^{p+1} g_w; g_w trained on residual vs a tight reference solver (or trajectory fitting). Local error O(delta eps^{p+1}); only local bound proved; overhead shrinks with base order; generalises to unseen step sizes empirically; tested on neural ODE/CNF, not N-body | FT |
| Um, Brand, Fei, Holl, Thuerey 2020, NeurIPS, 2007.00016 (solver-in-the-loop) | Learned correction to coarse solver trained with solver in the loop is stable for hundreds of steps and beats supervised-only | AB |
| Brandstetter, Worrall, Welling 2022, ICLR, 2202.03376 | Pushforward trick and temporal bundling against rollout drift | AB (already in Augur's rollout training in spirit) |
| Sanchez-Gonzalez+ 2019, 1909.12790 (Hamiltonian graph networks + ODE integrators) | Zero-shot generalisation to unseen step sizes/integrator orders; with low-order integrator and coarse step both learned and true Hamiltonian struggle | AB |
| Chen, Zhang, Arjovsky, Bottou 2020 (Symplectic RNN), 1909.13334 | Training and testing with same integrator and step lets the net learn a modified Hamiltonian that offsets discretisation error | AB |
| Offen & Ober-Blobaum, 2108.02492 (inverse modified Hamiltonians); 2504.05493 (learned integration-error term with embedded RK safeguard) | Learn/remove integrator error as a Hamiltonian correction; the second uses an embedded RK pair as safeguard | AB (cited via root-cause doc) |
| Cai, Portegies Zwart, Podareanu 2021, NeurIPS workshop, 2111.15631 | Wisdom-Holman style: analytic two-body part + neural interaction part; 3-body over 1e5 steps, O(N) claimed | AB |
| Horn, Saz Ulibarrena, Koren, Portegies Zwart 2024, JCP, 2310.20398 | HNNs behave symplectically but hard to train and fail across wide input scales; hybrid falls back to numerical solver when net output looks inaccurate | AB |
| Janik & Benner 2025, 2509.20212 (T-HenonNets) | Symplectic-by-design learned maps that accept adaptive steps; universality only for separable H | AB (fetched) |
| Bigi, Ceriotti (+Spies) 2025/26, 2508.01068, PRL 136, 237301 | Learn the mechanical action: symplectic, time-reversible long-step map; applicable iteratively as correction to cheaper direct predictors; plain ML long-step predictors lose energy conservation and equipartition | AB |
| Shankar 2026, 2607.20235 | Levi-Civita regularised analytic baseline stable at e=.99; neural residuals and small MLPs failed to approach it, O(1) rollout energy error; authors call it a falsification study | AB (fetched) |
| Li, Yang, Allen-Blanchette 2026, 2603.06354 (FS-HNN) | Fast and slow Hamiltonian networks trained at different timescales to beat spectral bias | AB (fetched) |
| Dellnitz+ 2023, SISC 45, A579, 2104.03562 | RL time-step controllers beat classical adaptive schemes on chaotic systems; low-dimensional ODEs only | AB |
| Behrmann, Schramowski, Kersting 2019 (Meta-learning Runge-Kutta) | Learned step-size controller | AB (id not verified) |
| Klein+ 2023, NeurIPS, 2302.01170 (Timewarp) | Flow proposal inside MCMC jumps 1e5-1e6 fs; a sampler, not a trajectory integrator | AB; not applicable |
| Sanchez-Gonzalez+ 2020, ICML, 2002.09405 (GNS) | Rollout stability driven by message-passing depth and training noise; no substep study found | AB |

Observations: no paper found that studies how a learned particle simulator's rollout error depends on the number of substeps per tick or that trains with substeps unrolled (unrolling is Solver-in-the-loop applied to the integrator). The only evidence about learned forces absorbing integrator error is the Symplectic RNN statement plus Augur's own pair probe.

## 3. Adaptive compute and routing

| Ref | Finding | Tag |
|---|---|---|
| Raposo+ 2024, DeepMind, 2404.02258 (Mixture-of-Depths) | Fixed-capacity top-k routing per layer; skipped tokens bypass block; static tensor shapes; matches baseline at equal FLOPs, up to 50% faster sampling | AB |
| Graves 2016, 1603.08983 (ACT); Spinelli+ 2020, 2002.10306 (APGCN) | Learned halting; per-node variable number of message-passing rounds | AB |
| Early-Exit GNNs, 2505.18088 | Early-exit for graph nets | AB (title only) |
| Pfaff+ 2021, ICLR, 2010.03409 (MeshGraphNets) | Learned simulator with remeshing in rollout; 1-2 orders faster than reference sim | AB |
| ROBIN, 2506.06045 | Hierarchical GNN on algebraic-multigrid coarsening; multiscale, not adaptive | AB |
| Fan+ 2023, 2310.15153 | GNN with adaptive remeshing and larger timesteps as microstructure coarsens; time step global | AB |
| Fidkowski group 2023 (HRLES) | NN error indicator drives adaptive polynomial order in DG | AB (title/snippet) |

Meaning for Augur: these choose where to spend network FLOPs. Augur's problem is numerical (one force per tick), not capacity. The fixed grid cannot be remeshed without breaking the FFT. MoE/ACT would route U-Net compute, which does not help the integrator. Only MoD's fixed-capacity top-k pattern transfers (design 2). Conflicts: none with constraints, but irrelevant to the floor.

## 4. Multiple time stepping in MD and learned MD

| Ref | Finding | Tag |
|---|---|---|
| Cattin, Ple, Adjoua, Gouraud, Lagardere, Piquemal 2025, 2510.06562 (DMTS) | RESPA-style (BAOAB-RESPA) with expensive foundation NNP and a distilled cheap NNP (3.5 A vs 11 A receptive field, ~10x faster); expensive-minus-cheap force applied every 2-6 steps. ~4x speedup; outer step 5-6 fs with hydrogen mass repartitioning; water resonates with O-H overtone at 3 fs without HMR; 7 fs diverged. Protein instability came from long tail (>150 kcal/mol/A) in cheap-model errors, fixed by active learning (~400 ps), not by resonance avoidance. Hydration free energy MAE ~.09-.10 kcal/mol vs single-step | FT |
| DMTS-NC 2026 follow-up (arXiv id 2602.14975, per search) | Non-conservative cheap force more stable and 15-30% faster; outer step to 10 fs with HMR and friction | AB (search snippet) |
| Tuckerman, Berne, Martyna 1992 (RESPA) | Reversible multiple time scale MD | MEM |

Meaning for Augur: exactly design 1 with mesh = slow, pair = fast. Direct support that non-conservative fast forces are acceptable (soft prior OK). Warning: error tails of the fast force dominate failures.

## 5. Where will the coarse step fail (error estimators)

| Ref | Finding | Tag |
|---|---|---|
| Embedded / Richardson pairs (Hairer-Norsett-Wanner) | Difference between 1-step and 2-half-step solutions estimates local error | MEM |
| Horn+ 2024 (2310.20398) | Run numerical solver when net output looks inaccurate | AB |
| 2504.05493 | Embedded RK as safeguard for learned error term | AB |
| Hernandez & Dehnen 2023 (2301.06253) | Check a time-symmetric selection condition each step; redo violating steps (a few percent of steps) | AB |
| PRDP 2502.19611; INC 2511.12764 | Learned correctors for hybrid PDE solvers | AB (snippets only) |

No paper found that trains a classifier to predict coarse-step failure for particle systems. Augur has a free label: |Verlet-1 minus Verlet-2 (or 4-substep) update| per body from truth data, plus strong features (nn, local accel, speed, relative speed of nearest neighbour, tau_ij/dt).

## Ranked candidate designs for Augur

Each: mechanism; cost vs density; constraints. Costs are reasoned, not measured.

**1. Pair-local multi-rate substeps, mesh impulse frozen (RESPA/HOLD/SyMBA transplant).** Per tick: compute mesh force once; for bodies with symmetric pair criterion tau_ij < dt (nn<1 bodies are 18-30% by the root-cause table), run 2 or 4 KDK substeps on the aligned truth grid using only the near-field pair term (pp/knn, analytic P3M complement and any learned pair head), holding the mesh acceleration constant (optionally linearly extrapolated). Pairwise symmetric kicks keep momentum. Cost: flagged fraction x neighbours x substeps; O(N) with rung cap; grows ~sqrt(rho) in dense cores. Constraints: O(N) yes if capped; tileable yes (pair-local); interpretable yes; no hard-coded conservative force (pair term unchanged); generalises across N. Risks: resonance of fixed-frozen mesh over a tick in dense cores; substep pair term must match full-model near field (DMTS lesson); retraining needed, since E absorbs part of the floor and fine-tuning E fails dE rows (E1), so from-scratch (pairfull-type) base. First test (no training): exact forces with mesh/pair split, pair substepped only for flagged bodies, compare err@20/50/100 to Verlet-1, global 2-substep, truth. If it reaches the 2-substep numbers, the design is viable; if not, the floor is mesh-force time variation, not pair.

**2. Fixed-budget top-k selector with embedded-pair error score (MoD pattern + embedded estimate).** Score each body by predicted or computed |Verlet-1 - Verlet-2| (cheap proxy: tau_ij/dt, or a tiny MLP on nn, |a|, v_rel trained on truth labels); take top fraction c of bodies per tile to receive design-1 substeps. Cost: bounded at c N regardless of density (static shapes, GPU friendly). Constraints: O(N) strict; tileable if top-k is per tile; interpretable (score = timestep ratio, inspectable); reversibility: selection depends on state, so use symmetric scores and keep the set fixed within a tick. Risk: dense cores exceed budget and silently lose accuracy, so report the overflow fraction. Use as the selector for 1; start with the classical criterion, learn only if it fails.

**3. Hypersolver-style residual on the pair step.** After Verlet-1, add eps^3-scaled learned correction g(pair features, dt) trained on single-step residuals against the 4-substep truth for flagged bodies (residual fitting, FT Poli+). No extra force evaluations; dense local label, no chaotic tail in the loss; checks cause #3 hypothesis cheaply (does a local-label loss avoid the bulk-for-tail trade). Cost O(N * knn) on flagged bodies. Constraints: no hard-coded conservation, soft energy prior possible; interpretable as "integrator-error term" separable from the force; generalises poorly if dt changes (only local bound proved). Risk: overlaps with what the pair head already absorbs, giving double-counting; non-symplectic drift on dE rows. Run after 1 shows the floor is pair-local.

**4. Hybrid high-order sub-integrator for close components.** Connected components of the timestep graph (Janes+ 2014) with a high-order or recursively subdivided integrator (IAS15-like or Gauss-Legendre, SyMBA-style) for the flagged component, mesh frozen. Per-component independence = embarrassingly parallel and tileable. Cost: per component O(n_c^2 * order); bounded if component size capped, otherwise explodes in clusters (a cluster core merges into one giant component; this is the main risk). Constraints: interpretable; closest to classical exactness; conflicts with O(N) in dense cores unless component size capped. Rank 4 because the rung-subdivision (1) covers most of the gain at lower complexity.

**5. Global 2-substep KDK (control and fallback).** Known to cut the exact-force floor 5-7x (root-cause table); cost 2x force evals (mesh included) or ~1x mesh + 2x pair if mesh frozen, which makes it design 1 with every body flagged. Not "only where needed" but required as the control and the simplest ship option; also tells whether the frozen-mesh approximation is harmless (compare to true 2-mesh-eval substep). Untested on trained E, per brief.

**6. Learned structure-preserving pair map (SympNet/HenonNet/learned action; Cai+ two-body + neural interaction).** Replace the Verlet pair step with a learned symplectic, time-reversible map that takes step size as input. Cost O(N * knn). Constraints: preserves structure by design but conflicts with interpretability and the "no hard-coded conservative forces" intent (symplectic by construction is a hard structural prior, arguably the same as a conservative force); evidence of fragility (Horn+, Shankar 2026); no N-body-at-scale demonstration found. Lowest priority; revisit only if 1-4 fail.

Not recommended (reasons):
- Higher-order global schemes (Yoshida 4th, Forest-Ruth, Hermite, IAS15 everywhere): need >=3 force evaluations or force derivatives; with 6% learned force noise the extra order is wasted, and negative substeps are hostile to the mesh pipeline (Pelupessy FT: negative substeps enlarge interval). Only worth it on exact analytic pair forces, which is design 4.
- Time-transformed / regularised schemes (Preto-Tremaine, KS, Levi-Civita): needed for unsoftened 1/r singularities; Augur's kernel is softened, truth has fixed 4 substeps.
- RL / learned step-size controllers (Dellnitz+, Behrmann+): validated on small ODEs; classical criteria already suffice; extra training risk.
- ACT, MoE, early exit, remeshing/AMR GNNs: spend network compute, not integrator compute (see section 3).
- Parareal and learned relabel: already rejected in plan doc.

## Suggested experiment order (for the experimenter)
1. Oracle (exact force) decomposition: Verlet-1, pair-only substeps on flagged set (nn<.5, <1, tau_ij<dt), global 2-substep with mesh re-evaluated vs frozen; report err@20/50/100 and dE for expB and clusters. Sizes the ceiling and tests the frozen-mesh assumption.
2. Histogram of rung assignments under the symmetric criterion in expB, cluster, a 1e3+ body dense scene: gives the cost curve vs density that this survey could not measure.
3. Correlation of per-body |Verlet-1 - 4-substep| with nn, tau_ij/dt, |a|: tells whether a classical score suffices before building a learned selector.
4. Then training (from scratch on the pairfull base, substeps unrolled in the loss, per the solver-in-the-loop pattern), one change at a time against a frozen baseline.

## Open flags
- Nothing here was verified against Augur code; the cost-vs-density statements are analytical expectations.
- Several papers are dated 2025-2026 and were seen only via search/fetch summaries.
- PDFs could not be read locally (no PDF tooling); FT tags mean WebFetch's summarised HTML text.
- The claim that Augur's pair term can be re-evaluated per substep without the mesh assumes the learned near-field heads (kmlp/ppmlp) take only pair features and do not read the U-Net feature map at the body; verify in code.
