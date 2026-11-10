# Architecture push: twenty broad experiments (design spec, 2026-11-09)

Status: DRAFT for user approval. Nothing here has been run, scripted or queued. No pass/kill line below is registered until the user approves it and it is copied into `docs/debugging/experiment-log.md`.

## 1. Intent

The last three weeks of work on the scatter field and the tiered hierarchy were mostly loss terms, fine-tunes, seeds and probes on one fixed architecture. Most were KILL. This spec proposes twenty experiments that each change a structural part of the model (how space is discretised, how the far field is computed, how the near field is computed, how the force law enters, how time and memory are handled, and how readable the model is), so that the next round tests architecture rather than recipe.

## 2. Definitions (first use)

- Tick: one simulation step (dt 0.1). Body: one particle.
- err@k: mean position error per body after k self-fed ticks against the exact simulation. dE@k: relative energy drift at tick k, compared with the truth energy at the same tick E(t), never E(0).
- Scatter field: the current model family. Bodies are deposited on a grid (CIC, cloud-in-cell, bilinear spreading), a U-Net (multi-level convolutional network) acts on the grid, forces are read back at each body. It also has an FFT kernel (a learned radial far-field kernel applied by fast Fourier transform) and a pair term (a small learned network over the 16 nearest neighbours inside radius 2; "pp=2, knn=16").
- Exp E: the pinned single-law gravity checkpoint `ms_kp_pot_v_g128` (grid 128, extent 64, unit mass).
- Tier 0: the finest tier of the tiered hierarchy, a scatter field run in 32-unit tiles with a 16-unit halo (overlap read from neighbours). BL: the tiled baseline that tier-0 results are divided by.
- Replay checkpoint: tier 0 after visited-state label replay (job 4089), the best tiled model so far.
- Mixed-law model: `MultiScatterField`, trained on six force laws (pow1, pow2, pow3, yuk8, yuk128, lj; pow1.5 and yuk32 held out) with a 20-number law descriptor "theta". Checkpoints main_s4738 and main_s4739.
- Four requirements: generalization (any particle system, any rules), tileable, O(N) cost (linear in body count N), interpretability (rules recoverable from the model).
- GPU-h: GPU-hours on one Polaris H200.

## 3. Where the architecture stands (sources checked 2026-11-09)

| Axis | Best result | Open failure | Source |
| --- | --- | --- | --- |
| Single-law gravity | Exp E; 1e9 bodies at 1.33-1.39 s/tick, 1e10 at 13.4 s/tick | accuracy holds about 20 ticks at 1e5 bodies (err@20 1.33, err@100 18.9); unit mass only; planar only | architectures.md:114-115 |
| Tiled tier 0 | err@20 0.70-0.77x BL after replay (one base seed; second base 0.81-0.95x); the closed-loop model before replay passes the err@20 line on 4 of 4 training seeds | dE@100 still 1.78-2.29x BL; gain sits in dense bodies (at least 20 neighbours: 89-92% of the error reduction from 33% of bodies) | supervisor-log 10-29 16:00; experiment-log.md:3625-3655, 3778-3781 |
| Mixed-law | main_s4739 gravity err@20 0.106 | Exp E is 0.0407; main_s4738 7.89 with a dead decoder level; held-out pow1.5 at 1.6-2.5x its single-law comparator (held-out yuk32 is 0.39-0.45x, against a weak comparator); amplitude-cap retrain KILL (33.08) | unexplored-directions-review-2026-11-09.md:10-11; experiment-log.md:3435 |
| Sub-cell behaviour | pair term covers r below 2; its collapsed-cloud slowdown at 3e6 bodies was fixed by the adaptive-depth change (job 7235) | half-cell shift changes force 3.7-24.5% on 6 of 6 laws for the mixed checkpoints, 6-10% for Exp E; averaging four shifted passes of frozen Exp E at inference cut err@20 39% at 4x cost (dE not measured) | supervisor-log.md:274; error-correction-plan.md:53, 78; experiment-log.md:3765-3769 |
| Hierarchy | upper tiers add 4-10% error over an exact external field | end-to-end KILL (1.5-1.7x BL); separate weights per tier; frozen sequential training | architectures.md:51-67 |
| Interpretability | exponents recovered for pow2 and the long Yukawa law on one checkpoint; pow1 exponent error 0.108 against a 0.10 line | rule-free recovery KILL (2 of 16 cells); dictionary features only causal from 5 features up; single-law sizing INCONCLUSIVE (widths 8 and 16 qualify, 64 not better) | experiment-log.md:3477-3478; rule-free-force-recovery.md; supervisor-log.md:269 |
| Chaos | none | err@100 is below the reference noise floor in black-hole scenes, so long-horizon accuracy has no usable metric | architectures.md:130 |

Outside literature checked (web search, abstracts and summaries only, not full text; treat as leads): Erwin, a ball-tree hierarchical transformer with linear-time attention for cosmology and molecular dynamics (arXiv 2502.17019, ICML 2025); a learned fast-multipole network for Green's operators (arXiv 2509.20591); SOG-Net, long-range forces as a sum of Gaussians (arXiv 2502.04668); symbolic force-law recovery from graph-network messages (arXiv 1909.05862); PDE-Refiner and training-noise methods for rollout stability (arXiv 2308.05732); a 2026 report that symplectic structure buys stability but not transfer to an unseen law (arXiv 2609.19674). None reports linear cost at 1e9 bodies with tiling, so there is no outside state of the art to copy directly; the ideas below borrow mechanisms, not results.

## 4. Goals, non-goals, anti-goals

Goals
- Twenty experiments, each one structural change against a named frozen control, each scored on the four requirements.
- Each has a cheap first stage (under 1.5 GPU-h) that gates the full run, so a dead idea costs little.
- Lines are ratios against a control re-run in the same job on the same held-out seeds, so no number depends on an old evaluation set. The exception is where an existing pre-registered line with absolute values is reused unchanged (experiments 10, 11, 15).

Non-goals
- No loss-term, optimizer, seed-count or schedule experiments (already-tried lists: architectures.md "Verified cons", audit appendix).
- No neuron-graph work (paused; reproduction FAIL, experiment-log.md:3827).
- No contact or computational-fluid-dynamics line (user ruling).
- No implementation plan. That follows approval, one plan per experiment.

Anti-goals (a result that would count as failure even if numbers improve)
- Any gain bought by an analytic force law, a force table, or a larger exact pair radius.
- Any gain bought by conservation that is hard-wired (potential-only output, forced antisymmetric pair force). Soft penalties and zero-initialised residuals are allowed.
- Any change whose cost grows faster than linearly with body count or with local density.
- A bundle of several changes with no ablation.

## 5. Constraints every experiment satisfies

1. Four requirements stated per experiment (section 7 scorecard column).
2. pp=2, knn=16 stays in every scatter-field experiment. Experiment 4 is a different model family with no pair term (flagged, section 10). Cost scaling with density is stated.
3. No hard-wired conservation, and energy terms match the truth's energy at the same tick, never zero drift. Three experiments sit near these rules and are flagged: 7 (solve step), 13 (soft power-matching penalty), 15 (rotation symmetry). Each is opt-in and needs the user's call (section 10).
4. One change per run against the frozen control; matched training budget (the mid-density "bias" on 10-26 was undertraining, so unmatched budgets are not accepted).
5. No sweeps. Where a design has a free number, one value is fixed here or by the builder before launch.
6. Seed 4738 first. Held-out evaluation seeds 9100/9200/9300, then a fresh set 12000/12100/12200 before any result is called real. A second training seed (4739) only after a pass.
7. At most 2 GPUs at once across everything; jobs chained by dependency; no node pinning.
8. Any job over 1 hour: full-state checkpoints, resume, SIGTERM handling, confirmed in the runner before launch.
9. Every run that produces rollouts gets the diagnostic grid render and a fresh, unprimed reviewer read before reporting. Every numeric result gets a fresh reviewer who recomputes the headline number and is given the pass/kill line.
10. Every outcome, including failures, is logged in `docs/debugging/experiment-log.md` when it finishes. Names are descriptive kebab-case as written below.
11. Standard length is 50 epochs where the runner counts epochs; iteration-based runners keep the control's iteration count (needs-user item 2).

## 6. Shared scorecard

Controls
- G: Exp E, single-law gravity.
- T: the replay checkpoint (tier 0 tiled, job 4089 lineage).
- M: main_s4739, mixed-law.
- S: single-law gravity at width 8 (the smallest unambiguous width from job 7256), for the interpretability experiments.

Metrics (all as candidate divided by control, same job, same seeds)
- err@20 and err@5 (short-horizon accuracy).
- dE@100 against truth E(t).
- Tiled-versus-single-grid force difference (tileability).
- Nanoseconds per body per tick at 1e5 and 1e6 bodies, and the fitted cost exponent (O(N)).
- Dense-bin error: late-tick (60-100) squared force error on bodies with at least 20 neighbours.
- Law metrics (mixed-law experiments): tick-1 force error per trained law, and on held-out pow1.5 and yuk32, each against its single-law comparator.
- Rule recovery (interpretability experiments): cells passed out of 16 on the rule-free recovery protocol, unchanged lines.

Default lines, used unless an experiment states its own
- PASS: err@20 at most 1.00x control AND the experiment's target metric meets its stated line AND cost at most 1.5x control.
- KILL: err@20 above 1.15x control (the existing acceptance gate) OR the target metric shows no gain (stated per experiment).
- Between: INCONCLUSIVE. No loosening after launch.

## 7. The twenty experiments

Fields: change; why (evidence); already-tried check; target line with a value that would fail it; cost (stage 1 / full, rough, from logged rates); requirements touched. "Gen/Til/O(N)/Int" are the four requirements.

### Group A. How space is discretised

**1. nested-refinement-levels-on-the-scatter-field-with-smooth-level-blend** (control G, then T)
- Change: keep the scatter field as it is (bilinear deposit, U-Net, kernel, pair term) and add two extra grid levels at half and quarter cell size, present only where a cell holds more than a fixed body count. Same U-Net weights at every level plus a level input; a body's force is a smooth blend between levels, not a switch.
- Why: the replay gain and the remaining drift both sit in dense bodies; the densest 10% of bodies hold 12-26% of squared error; architectures.md lists "no refinement tier has been built" as a structural limit.
- Already tried, and this is a re-test: `MultiLevelScatterField` (2026-10-01) was a from-scratch, density-adaptive, block-sparse grid. It lost 3-10x to the dense model and was unstable (scatter-field-architecture.md:44-54, 85). That version used nearest-cell deposit with a per-cell linear expansion, hard switching between levels, a trainable mass-change channel (the cause of the instability) and no pair term, at 4000 iterations with loss still falling. Its own entry lists as not tested: longer training, bilinear deposit instead of nearest-cell, a smooth blend between levels. This experiment is exactly those three, bundled because they are coupled (a blend needs overlapping deposits); the mass-change channel is dropped. If it passes, each of the three is ablated. The scoped frozen-base residual tier (audit row 20) never ran.
- Line: PASS dense-bin error at most 0.70x control AND dE@100 at most 0.75x AND err@20 at most 1.00x AND cost at most 1.5x. KILL dense-bin above 0.90x or err@20 above 1.15x. Fails pass: dense-bin 0.80x.
- Cost: 1 / 6 GPU-h. Requirements: Gen (dense scenes), O(N) (refined cells are bounded per level; state the count), Til (patch halos must match tile halos).

**2. summary-token-exchange-between-neighbour-tiles** (control T)
- Change: each tier-0 tile writes a short learned summary vector each tick; a tile reads its eight neighbours' summaries (and its own from the previous tick) as extra input channels. Cost is constant per tile.
- Why: exchanging the recurrent field in the halo was the largest single gain in the tiered line (tiled err@20 from 1.66-1.82x to 0.75-0.82x BL). The user deferred richer tile-to-tile communication on 2026-10-24 (heterogeneous-tiered.md:158) and it was never dispatched (audit row 29).
- Already tried: halo overlap and halo exchange of the recurrent field only.
- Line (audit row 29, with a drift guard added): PASS tiled err@20 at most 0.90x control AND dE@100 at most 1.00x control AND cost at most 1.1x. KILL err@20 at or above 0.97x. Fails pass: 0.94x.
- Cost: 0.5 (measure how much of the control's force error is predictable from neighbour-tile state, inference only) / 3 GPU-h. Requirements: Til, O(N).

**3. staggered-double-grid-with-quadratic-deposit** (control G)
- Change: replace bilinear deposit and gather with a quadratic (three-cell) shape, and run two grids offset by half a cell through the same U-Net, averaging the gathered force. Trained that way, not only run that way.
- Why: the model is not shift-consistent below a cell (half-cell shift changes force 6-10% on Exp E, 3.7-24.5% on the mixed checkpoints). Averaging four shifted passes of frozen Exp E at inference already cut err@20 39% and err@100 30%, at 4x cost and with dE not measured (error-correction-plan.md:78). Grid-origin jitter as data augmentation raised dE@1000 5.8x, so the data-side fix is closed. The pending shift-consistency loss is a loss term; this is the structural version, aiming for most of the four-pass gain at about 2x cost.
- Already tried: the inference-only four-pass average above. A trained staggered or quadratic deposit never ran (scatter-correction-research.md:64, 130, marked untested).
- Line: PASS err@20 at most 0.75x control AND shift-probe value at most 0.50x control AND dE@100 at most 1.10x AND cost at most 2.2x. KILL err@20 above 0.95x or dE@100 above 1.5x. Fails pass: err@20 0.85x.
- Cost: 0.3 (two-pass and four-pass averaging of frozen Exp E with dE measured, which fills the gap in the earlier result) / 3 GPU-h. Requirements: Til (seams are sub-cell shifts), Gen.

**4. ball-tree-particle-attention-tier-0** (control G)
- Change: no grid. Bodies are sorted into a ball tree; attention runs inside fixed-size balls, with coarsening and refinement between tree levels (the Erwin layout). Force is read per body.
- Why: every sub-cell problem above comes from the grid. The earlier particle model (token-per-ball) failed on gravity at 100-1000 bodies because a fixed-radius neighbour graph has no long range; a tree gives long range at linear cost.
- Already tried: no tree or hierarchical attention model over bodies in the notes (search of docs for ball tree, Erwin, point transformer: zero hits). Windowed attention was tried only on a rendered image grid (checkerboard artifact, removed at commit f0e709b), a different model.
- Line: PASS err@20 at most 1.15x G at N 1000 AND cost exponent at most 1.10 from 1e3 to 1e6 AND at most 100 ns per body per tick at 1e6. KILL err@20 above 2.0x or exponent above 1.25. Fails pass: err@20 1.4x.
- Cost: 1.5 / 8 GPU-h. Risk: tree partitions are not translation-invariant, so tiling may fail; report tiled-versus-global difference without a line (curiosity for that metric). Requirements: Gen, O(N); Til at risk.

**5. three-dimensional-sparse-voxel-scatter-field** (control: 2D G at equal N)
- Change: the scatter field in 3D, storing only occupied voxels plus a halo.
- Why: everything shown so far is planar; audit row 33 never ran because a dense 3D grid costs 8x.
- Already tried: no 3D grid code (audit row 33).
- Line (audit row 33's two values, with the err@20 kill edge added here): PASS 3D Plummer-sphere err@20 within 2x the 2D ratio to exact at equal N AND cost at most 12x the 2D model. KILL cost above 12x or err@20 above 4x. Fails pass: 3x.
- Cost: 1.5 / 20 GPU-h (largest item; checkpoints required). Requirements: Gen, O(N).

### Group B. How the far field is computed

**6. gaussian-ladder-far-kernel-replacing-fft** (control G; then M)
- Change: the far kernel becomes a sum of Gaussians whose widths double from level to level; each is applied as a small blur on a grid coarsened to its width. No FFT. Amplitudes are free in sign and, in the mixed-law version, a function of theta.
- Why: the FFT kernel is global, so tiled inference needs a separate coarse global far term; a ladder of local blurs is tile-local by construction. The amplitudes are directly readable: a power law is a straight line in log amplitude against level, a screened law is a cut-off. The earlier theta-conditioned and inferred radial kernels were KILL (commit b2a0522); this changes the kernel's form, not its conditioning.
- Already tried: zero hits for sum of Gaussians in the notes.
- Line: PASS err@20 at most 1.15x G AND tiled-versus-single-grid force difference at most 4% with no global term AND exponent read from the amplitudes within 10% on pow2. KILL err@20 above 1.30x. Fails pass: exponent error 15%.
- Cost: 0.5 (fit the ladder to the frozen learned kernel, CPU plus short GPU check) / 3 GPU-h. Requirements: Til, Int, O(N).

**7. learned-local-operator-with-repeated-solve-for-the-longest-range-law** (control: the pow1 single-law comparator)
- Change: instead of learning the far kernel, learn a small local stencil (an operator on the potential) and solve "operator applied to potential equals density" with a few weight-shared multigrid cycles. The U-Net remains as a zero-initialised residual.
- Why: the laws are defined by force exponent (powP has force about r to the minus P; the Yukawa laws are the pow2 force times an exponential cut-off). In the plane, the Laplacian's solution gives a 1/r force, which is pow1, the law the mixed model handles worst (41 of 48 BAD scenes; exponent error 0.108 against a 0.10 line). pow1 has the longest range, which a convolution with finite support cannot hold and a solve can.
- Scope limit, stated up front: only pow1 is expected to be the solution of a local operator in 2D. pow2, pow3 and these Yukawa forms are not (the 2D screened operator gives a different functional form). So this is a one-law experiment and scores low on generalization by design. Stage 1 confirms the scope analytically on CPU.
- Already tried: zero hits for multigrid or learned operators.
- Line: PASS pow1 tick-1 force error at most 0.70x the pow1 single-law comparator AND pow1 err@20 at most 0.70x that comparator AND the learned stencil within 10% of a scaled five-point Laplacian (seed 4738; seed 4739 after a pass). KILL force error above 1.0x. Fails pass: 0.85x.
- Cost: 0 / 3 GPU-h. Flag: a solve step is structure, though not a conservation law; user call. Requirements: Gen (one law only), Til (local), Int.

**8. one-network-for-all-tiers-with-scale-input** (control: per-tier-weights upper tiers)
- Change: one set of upper-tier weights applied at every tier, with the tier's length scale as an input, replacing separate weights per tier.
- Why: the hierarchy's depth is currently fixed by how many tiers were trained. Shared weights make the hierarchy extend to any span, which is tileability in scale; and how the output changes with the scale input is the force law's range behaviour, readable directly. The two-upper-tier span test (audit row 13) was scoped and never run.
- Already tried: no (architectures.md:42 "different between tiers").
- Line: train on spans up to 256, evaluate at 1024 (two unseen tiers). PASS far-field relative error at 1024 at most 1.2x its own value at 256 AND its value at 256 at most 1.0x the per-tier control at 256. KILL error at 1024 above 2.0x its own value at 256, or its value at 256 above 1.15x the control. Fails pass: 1.5x.
- Cost: 1 / 6 GPU-h. Needs the tiered trainer ported from branch tier0-tiled-closed-loop. Requirements: Til, O(N), Int.

### Group C. How the near field and the bodies are represented

**9. neighbour-context-pair-branch-replacing-central-pair-term** (control G, two seeds)
- Change: the pair network receives relative velocity and local density as well as distance, and outputs a free 2-vector (zero-initialised) instead of a radial magnitude.
- Why: removes the hard-wired central, velocity-independent pair force, in line with the no-hard-wired-conservation rule, and lets the model represent drag-like or uneven pair laws. Audit row 9, never run; the 11-09 review set it aside only because that review was limited to training and data changes.
- Already tried: a frozen-base residual with the same inputs gave one-step error -44% and no rollout gain; never trained into the model.
- Line (audit row 9): PASS err@20 at most 0.90x control on both seeds AND tangential output below 5% of radial on gravity data. KILL err@20 at or above 0.95x or momentum error above control. Fails pass: 0.93x.
- Cost: 0.5 / 2 GPU-h. Requirements: Gen, Int.

**10. learned-per-body-property-channels** (control G; stage 1 is the existing mass design)
- Change: each body carries a small property vector deposited on its own grid channels. Stage 1 trains the already-written mass-conditioned design (property = given mass). The full run replaces the given mass with a vector inferred from the body's own recent motion.
- Why: unit mass is the largest generalization gap (1000:1 mass ratio gave binary err@2 216, or 2.3 when the time step is matched; the planetesimal disk is out of range). The mass design, curriculum and 10 CPU tests exist and were never trained (audit row 12). An inferred vector also covers species and charge sign without a new input per property.
- Already tried: mass fine-tunes of the gravity model failed in-distribution by 1.2-3.9x; from-scratch conditioning never ran.
- Line: stage 1 uses the pre-registered mass lines (planet semi-major-axis error at most 0.02, in-distribution gate 1.15x; KILL above 0.15 or 1.5x). Full run: PASS same lines without mass given AND rank correlation of the inferred property with true mass at least 0.9. Fails pass: correlation 0.8.
- Cost: 2.5 / 6 GPU-h. Requirements: Gen, Int.

### Group D. How the force law enters

**11. law-blind-unet-trunk** (control M)
- Change: theta no longer enters the U-Net. The far kernel and the pair network stay conditioned on theta exactly as now; the U-Net sees only deposited fields and the kernel's output. Stage 1 is the never-run cheaper variant: Exp E frozen, with small per-law kernel and pair heads (audit row 16).
- Why: in main_s4738 one decoder level is dead and another has activations near 3700; the amplitude-cap retrain made gravity worse (33.08). The U-Net is being asked to change behaviour with theta. If laws differ only in the two readable parts, the trunk cannot be destabilised by one law, and the law lives where it can be read.
- Already tried: no model without theta in the U-Net; row 16 never run. Related KILL to keep in view: a theta-conditioned far kernel alone failed on held-out laws (commit b2a0522, yuk32 3.4x), so the kernel's conditioning is a known weak point that this experiment does not change.
- Line: stage 1 (row 16): gravity err@20 at most 1.05x Exp E AND other laws within 2x their single-law comparators; KILL gravity above 1.15x. Full: PASS gravity err@20 at most 0.0509 on seeds 4738 and 4739 AND no dead decoder level AND every trained law's tick-1 force within 1.25x of M. KILL gravity above 0.106. Fails pass: 0.07.
- Cost: 3.5 / 7.5 GPU-h. Requirements: Gen, Int.

**12. law-inferred-from-motion-on-random-law-zoo** (control: experiment 11's model, or M if 11 is KILL)
- Change: no theta at all. An encoder reads a few ticks of observed motion and produces a short law code that takes theta's place as the input to the kernel and pair networks. Training laws are sampled from a generator of smooth radial laws (hundreds), not six.
- Why: "any particle system, even under different rules" cannot depend on a hand-written descriptor. Held-out pow1.5 sits at 1.6-2.5x its comparator and the log names more law diversity or a structural prior as the untested routes. The law code is also the recovered rule.
- Already tried: the inferred far kernel was KILL on all held-out laws (b2a0522), but it inferred only a radial kernel from six laws. This infers a code for the whole law path from a wide law family. The difference is real but the risk is the same; expected pass chance is low (about 20%, my estimate).
- Line: PASS held-out pow1.5 tick-1 force at most 1.3x its single-law comparator AND held-out yuk32 at most 1.0x the control's own yuk32 error (the control already beats the yuk32 comparator) AND the code predicts the true exponent with R-squared at least 0.9 on held-out laws. KILL pow1.5 above 1.6x or yuk32 above 1.3x the control. Fails pass: pow1.5 1.45x.
- Cost: 1 / 9 GPU-h. Requirements: Gen, Int.

### Group E. How time, memory and symmetry are handled

**13. separated-velocity-branch-with-soft-power-matching** (control T)
- Change: tier 0 is split into a position-only force path and a second branch that sees velocity and the recurrent field. The work each branch does per tile is then measurable separately, and a soft term pulls the model's total power per tile toward the truth's power per tile at the same tick (a similarity target, not zero work).
- Why: velocity and history inputs are what leak energy (the molecular-dynamics mesh drifted -0.26/-0.29 with them, at most 0.013 with those inputs zeroed, and -0.009 with the mesh off entirely), yet removing them on gravity was KILL (err@20 1.56-1.85x) although one-step error barely moved. They help rollouts and hurt energy at once; separating them lets each be measured and constrained alone. The held curl-and-drag penalty (audit row 22) acts on the total force and cannot tell the two apart.
- Already tried: input removal (KILL), total-force loss terms (KILL). No split.
- Line: PASS dE@100 at most 0.75x control AND err@20 at most 1.05x. KILL dE@100 at or above 0.95x or err@20 above 1.15x. Fails pass: dE@100 0.85x. Because this bundles a split and a loss term, the split alone (no term) is run as the ablation arm in the same job.
- Cost: 0.3 (measure the work done by each input group on the frozen model, inference only) / 6.5 GPU-h. Flag: soft energy prior, written as similarity to truth per the energy-term rule; opt-in. Requirements: Gen (flag), Til.

**14. sampled-force-tier-0-scored-on-distributions-past-the-chaos-horizon** (control G)
- Change: the model takes a noise input and is trained with a proper scoring rule over small ensembles, so it produces samples rather than a mean. Long rollouts are scored on distributions (radial density profile, velocity spread, energy) against an ensemble of slightly perturbed exact runs.
- Why: past the chaos horizon a mean-squared-error model is rewarded for regressing to the mean. The grid models blurred, and tier 0 over-contracts clumps at mid range. There is no usable long-horizon metric today. This gives both a model and a metric for ticks 100-1000.
- Already tried: the chaos-floor noise control was dropped unrun; no sampled-output model in the notes.
- Line: PASS at tick 300, at least 80% of radial bins of the density profile and of the velocity spread fall inside the exact ensemble's two-standard-deviation band AND ensemble-mean err@20 at most 1.10x control. KILL under 50% of bins or err@20 above 1.25x. Fails pass: 65%. Stage 1 scores the unchanged control on the same distribution metric; if the control already passes, the experiment is closed as unnecessary.
- Cost: 1 / 5 GPU-h. Requirements: Gen.

**15. rotation-steerable-unet-weights** (control G)
- Change: U-Net filters are built from rotation harmonics so the network is rotation-consistent in one pass.
- Why: the four-pass symmetrised U-Net cut core drift from 0.087 to 0.0015-0.0067 per tick on three seeds but cost 3-4x and was KILL on fresh seed 12300 (0.0165 against 0.0105). The 1e9 black-hole core still walks to y = 13.7 by tick 300. Built-in symmetry is finer than four-fold and costs one pass.
- Already tried: only the four-pass version.
- Line (the core-drift line of experiment-log.md:3620, unchanged, plus a cost clause): seeds 12100/12200/12300; gates are core drift at most 0.007 per tick, err@20 at most 1.05x stock, black-hole-scene dE@1000 at most 1.15x stock. PASS all gates on every seed AND pooled dE ratio at most 1.095 AND cost at most 1.3x. KILL any value above 1.5x its gate. Fails pass: drift 0.009 on one seed.
- Cost: 0.5 / 4 GPU-h. Flag: rotation symmetry is a prior that a uniform external field breaks; opt-in. Requirements: Gen (flag), Til.

**16. timestep-as-input-model** (control G)
- Change: dt becomes an input and training mixes several step sizes. One model then runs at any step, and dense tiles can take smaller steps than sparse ones.
- Why: retraining at half the step cut err@20 about 40% and drift 3.5x at 1.9x cost; the dt 0.05 fine-tune (audit row 6) is held and the inference-only half-step test cannot separate causes. A step-aware model answers that question and gives multi-rate stepping without a second model.
- Already tried: fixed-step retrains; substep routing and adaptive substeps on fixed-step weights (negative). No step-conditioned model.
- Line: PASS at dt 0.1, err@20 at most 1.05x control; at dt 0.05, err@20 at most 0.70x and dE@100 at most 0.50x of control at dt 0.1; step sizes not trained on (0.075) interpolate within 15% of the line between neighbours. KILL err@20 at dt 0.1 above 1.15x. Fails pass: 0.80x at dt 0.05.
- Cost: 1 (generate truth at three step sizes on GPU) / 5 GPU-h. Requirements: Gen, O(N) (cost per unit simulated time).

### Group F. Readability by construction

**17. two-number-message-bottleneck-single-law** (control S)
- Change: on single-law gravity, the pair network and the kernel each pass through a two-number bottleneck before the force is formed, and the U-Net residual's share of the force is reported. A symbolic fit is then run on the bottleneck values.
- Why: post-hoc recovery was KILL (2 of 16) and dictionary features were causal only from 5 features up. The user's plan is single law first with a network sized to the law. A bottleneck the size of a force vector makes the message be the force up to a linear map (the graph-network result, arXiv 1909.05862).
- Already tried: symbolic fitting only on curves extracted after training.
- Line: PASS linear fit of the bottleneck to the true pair force R-squared at least 0.95 AND exponent within 10% AND err@20 at most 1.15x control (training seed 4738; seeds 4739 and 4740 only after a pass) AND U-Net share of force variance reported. KILL R-squared below 0.80 or err@20 above 1.30x. Fails pass: R-squared 0.90.
- Cost: 0.7 / 2.5 GPU-h. Requirements: Int.

**18. sparse-code-unet-single-law** (control S and the width-32 model)
- Change: each U-Net level's activation is replaced, during training, by a code with at most 8 active features per cell.
- Why: the user's stated concern is that a law spreads over many layers and neurons. Training with sparse codes makes each level's features the model's actual variables rather than an after-the-fact dictionary.
- Already tried: dictionaries fitted after training only (curiosity runs, no verdict).
- Line: PASS held-out force error within 10% of the width-32 dense model (the user-approved sizing tolerance) AND removing the top 5 features raises median force error at least 5x the matched-random mean and above the random maximum on at least 2 of 3 held-out sets. KILL force error above 1.30x or ablation under 2x. Fails pass: ablation 3x.
- Cost: 0.7 / 2.5 GPU-h. Requirements: Int.

**19. bit-output-cell-model-at-64-grid-with-long-range-law** (control: its float twin)
- Change: the bit-output cell model (per-cell bits of the momentum change, plain bits with soft decode) moved from an 8x8 short-range toy to a 64x64 grid with a long-range law, with coarser copies of the grid as extra inputs.
- Why: plain bits passed on three seeds, and soft decode and Gray code each improved on plain bits on three seeds (Gray plus soft decode together is untested, so one is used here), but only at 8x8 with a short-range force; the open items in its own entry are larger grid and long range.
- Already tried: 8x8 short range only.
- Line (thresholds of the 11-08 pre-registration): PASS bits-to-float ratio at most 1.5 on force error and on rollout error at tick 20. KILL either above 3. Seed 4738; seeds 4739 and 4740 only after a pass. Fails pass: 2.0.
- Cost: 0.3 / 1.5 GPU-h. Requirements: Gen, Til.

**20. finite-speed-local-field-medium** (CURIOSITY, no verdict; needs user approval of the label)
- Change: the potential is a field with its own state, updated each tick by a learned local rule and sourced by the bodies, so influence travels cell to cell. No global operation anywhere.
- Why: this is the user's 09-28 idea; the two-body prototype exists (audit row 40) and was never taken past N = 2. It is the fully local limit of experiments 6 and 7.
- Already tried: two bodies only; the wave front contaminated the force for two crossing times.
- Observations to record: force error against exact at N = 2, 20 and 1000 as a function of propagation speed; cost per tick; whether a filtered front stays within 5% of the retarded-force reference at N = 2 and 20 (audit row 40's line, reported, not scored).
- Cost: 1 GPU-h. Requirements: Til, O(N), Gen.

## 8. Order and compute

Rough total: stage 1 for all twenty about 18 GPU-h; full runs about 112 GPU-h if every gate opens; in practice fewer. With 2 GPUs that is roughly 3 days of wall clock for everything, less after gates close.

Wave 1 (all stage 1 probes, about 18 GPU-h, two chains): slot 1: 13, 3, 2, 6, 9, 17, 18, 19, 14, 1. Slot 2: 7 (CPU), 16 data generation, 15, 10 (mass design), 11 (frozen-path heads), 8, 12, 4, 5, 20.

Wave 2 (full runs, only for experiments whose stage 1 did not close them), ordered by expected value per GPU-h (my ranking):
1. 3 staggered double grid (an inference-only version already cut err@20 39%)
2. 11 law-blind trunk (targets the located mixed-law failure)
3. 2 summary-token exchange between tiles (extends the largest tiered gain so far)
4. 6 Gaussian ladder (tileability and readability at once)
5. 17 and 18 (short; follow the agreed single-law plan)
6. 1 nested refinement levels (targets the located dense error, but re-tests a killed design)
7. 13 separated velocity branch
8. 10 learned per-body properties
9. 16, 15, 9, 8, 14
10. 12 (after 11), 4, 7 (one law only)
11. 5 last (largest)

After wave 2: one capstone that combines the passes, with one ablation per combined part. Not specified here; it depends on what passes.

## 9. Left to the builder

- Layer counts, channel widths and the single fixed value of any free number, chosen before launch and written into the pre-registration.
- Porting `train_scatter_multi.py` (branch near-field-retrain) and the tiered trainer (branch tier0-tiled-closed-loop) onto the working branch before anything depends on them.
- Confirming each "already tried" check again at launch time.
- One implementation plan per experiment after approval.

## 10. Needs the user

1. Approve, cut or reorder the twenty.
2. Training length: iteration-based runners are well under 50 epochs (80k iterations is about 17.8 epochs). Keep the control's iteration count (assumed here) or move to 50 epochs for both control and candidate?
3. Experiments 7, 13 and 15 add structure near the no-hard-wired-conservation rule (a solve step, a soft power-matching term, rotation symmetry). Allowed as opt-in?
4. Experiment 4 is a grid-free model with no pair term, which departs from "stay on pp=2, knn=16". Allowed as a test?
5. Experiment 20 as a curiosity run with no verdict?
6. Width for the single-law control S: 8 is assumed (smallest unambiguous in job 7256).
7. Experiment 1 re-tests a design that was killed on 2026-10-01, with the three causes its own entry left untested. Keep it, or drop it as already decided?
8. Experiment 7 covers one law of six. Keep it at low rank, or drop it?

## 11. Review

Independent review ran (fresh agent, 2026-11-09, read-only, against the notes, memory rules and git log). Verdict on the first draft: NOT CONFIRMED. All findings were applied in this version:
- A pair-term-free nested grid had already been run and killed (2026-10-01), and its cost motivation was already fixed by job 7235. That experiment was replaced by experiment 2 (tile summary exchange), and experiment 1 is now labelled a re-test.
- Experiment 7 misquoted the pow1 figure and wrongly claimed the Yukawa laws fit a local operator. It is now scoped to pow1 only.
- Experiment 3 missed an existing inference-only result (four-pass shift averaging, err@20 down 39%); it is now the evidence and the stage 1 design.
- Experiment 11 described theta conditioning of the kernel and pair networks as new; they already have it. The change is now only removing theta from the U-Net.
- Experiment 13 penalised work toward zero, against the energy-similarity rule; it now matches the truth's power.
- Lines for experiments 15 and 19 did not match the registered lines they cited; corrected. Smaller number and wording fixes in section 3 and experiments 5, 8, 10, 12, 17.

Confirmed by the reviewer as quoted correctly: the scale and err@k figures, replay 0.70-0.77 and 1.78-2.29, the dense-body share, 33.08, 3741, 41 of 48, the job 7256 widths, 2 of 16, seed 12300 at 0.0165 against 0.0105, jitter 5.8x, and audit rows 9, 12 and 16.

Not reviewed: the corrected text itself (no second review pass), the arXiv citations (search summaries only), every cost estimate, causal-field-2body.md (experiment 20), and the experiment log before line 3398.
