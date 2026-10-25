# Heterogeneous tiered architecture (theory notes)

Started 2026-10-24, theory mode (no code run). Motivated by the LJ scatter-field diagnostic results.

## Idea (user's, as confirmed)
- Stack of tiers instead of one fine tiled tier plus one global coarse tier.
- Tier k+1 tile governs a block of tier-k tiles (example: 2x2).
- Every tile, every tier: same architecture.
- Weights shared across all tiles within a tier (dynamics are position-independent).
- Weights differ between tiers (tier 0 handles particles directly; tier >= 1 handles tiles/aggregates).
- Each tile handles its own domain plus an edge overlap (halo); communicates with adjacent tiles at the same tier.
- Barnes-Hut analogy: near field resolved at fine tiers, far field arrives via coarse tiers.
- Tiers are layered as needed (depth chosen per problem).

## Tier 0 representation
- Same as current scatter field: particle properties stored as channels in a grid (scatter to grid, network on grid, gather to particles).
- Tier-0 tile = grid patch + halo.
- Unconfirmed inference: tier >= 1 input may be a coarsened grid of the same channels built from child tiles.

## Training regime (initial intent, may need testing)
- Train tier 0 first (particles); must handle particles crossing into its domain from neighbouring tiles.
- Freeze tier 0, train tier 1, repeat upward.

## Holes raised (2026-10-24, most severe first)
1. Does not address LJ failure (mesh injects non-conservative force); one learned field per tier multiplies the source. Needs soft potential-consistency / reciprocity prior per tier.
2. Gravity has no scale separation; coarse grid of mass/momentum channels carries only monopole/dipole, loses sub-cell structure (BH opening-angle problem). LJ is short-range and may not need tiers.
3. Interaction partition undefined: tier 0 halo and tier 1 child overlap both see near-field neighbours; need residual training or range-truncated targets (cutoff ill-defined for gravity).
4. Tier 0 trained alone vs full truth learns mean-field far-field bias, frozen in; tier 1 must cancel it.
5. Frozen sequential training: tier 1 trained on tier 0 one-step outputs misses tier 0 rollout errors; rollout-in-loop couples tiers; tier 0 cannot adapt.
6. Parent input from full child grids is not compressed; O(N) requires downsampling, which sets what information survives.
7. Overlapping parents give multiple corrections per particle; need partition-of-unity blending; seams at every tier.
8. Uniform tile hierarchy does not refine in dense regions or skip empty ones (multigrid/PM-like, not BH-like).
9. Overlap with existing in-tile U-Net coarse levels; benefit is bounded receptive field + per-scale weights, untested vs deeper single U-Net.
10. Per-tier timescales complicate coupling/conservation; domain-edge tiles break position-independent weight sharing.

## Fix attempts (walked one at a time)
### Hole 1 (proposed, pending user response)
- Each tier outputs scalar potential phi_k; force = -grad(phi_k) via matched scatter/gather kernels; sum over tiers is curl-free by construction.
- Plus small free residual force channel, softly penalised (curl, magnitude), so non-conserving systems stay representable.
- Mass-weighted mean force removed per tier over the closed domain (not per tile) -> exact total momentum conservation.
- Not fixed: learned phi=f(rho) is not a symmetric pair interaction (momentum balance comes from mean removal, not architecture); self-force needs scatter/gather adjointness; gradient stencil must match interpolation kernel.

### Hole 1 follow-ups
- Behaviors the hole-1 fix does not cover: non-reciprocal; external/open forces (mean removal would delete uniform gravity unless limited to the internal part); velocity-dependent (drag, Lorentz, Coriolis, speed cap); impulsive/dissipative contact; history; stochastic; angular momentum/torque; sub-cell stiff repulsion; mass exchange. Residual penalty (curl, magnitude) trains against the cases the residual is needed for.
- Tier application order: (a) all tiers read same state at t, outputs summed, order irrelevant (matches residual + frozen training; recommended); (b) tier k+1 reads tier k output, forced 0,1,2; (c) top-down 2,1,0 (FMM downward pass; tier 0 receives far field as input, helps hole 4, but tier-0-first training needs teacher far field -> train/test mismatch).
- Later tiers see only coarse aggregates, so can only fix smooth/tile-scale/global errors: uniform external force, net momentum, collective drag, thermostat, Coriolis, angular momentum drift, long-range non-reciprocal fields, wall far-field. Cannot fix per-particle/sub-cell: contact impulses, friction, stiff repulsion, speed cap, noise, history, mass exchange, torque, short-range non-reciprocity.

### Per-particle / sub-cell resolution (tier 0 only)
- Per-particle readout head: gather grid features + particle state -> shared MLP update.
- Per-particle hidden state carried across steps (history), can scatter as channels.
- Impulse output (delta-v) alongside force for contact/restitution.
- Speed cap by bounded output parameterization.
- Sub-cell moments (mean offset, second moments) in scatter; inherited by upper tiers.
- Bounded-neighbour local term (cell list, small fixed K); must respect the dropped pp=8 decision.
- Mass exchange as discrete events in the head.
- Tension with hole 1: head force is not a potential gradient; must sit in the soft-penalised residual channel; needs its own pair-symmetry/energy-drift penalty.
- Later holes pointing here: 2 (moments), 3 (near zone defines partition), 4 (truncated target), 6 (what survives coarsening), 8 (dense cores), 10 (contact sets step size); 7 only via particle ownership.

### Per-particle / sub-cell: user decisions (2026-10-24)
- Pursue: extra scatter channels (incl. explicit sub-cell moments) and a SMALL per-particle MLP head (head force goes in the soft-penalised residual channel).
- Dropped: impulse output (generalizes only to velocity); speed cap by parameterization (model should represent structure itself); bounded-neighbour term (violates pairwise prohibition).
- Deferred: mass exchange (interaction-type specific).
- Code check (scripts/scatter_field.py): current scatter is CIC of [m, m*vx, m*vy] only, bilinear gather; sub-cell position is implicit via corner weights; no explicit offset/second-moment channels; grid-level hidden medium state exists (hidden_ch), per-particle hidden state does not.

### Hole 2 (proposed, pending user response)
- Scatter explicit moments: mass, momentum, first-order offset moments, second-order moments (about each cell centre).
- Coarsening = exact moment translation (parallel-axis), lossless to chosen order, O(1) per child (FMM upward pass). Learned downsampler kept as an alternative to test.
- Each tier outputs local-expansion channels (phi, grad, Hessian) per coarse cell; children evaluate the Taylor expansion at their position; keeps -grad(phi_k) exact (hole 1).
- Each tier covers a distance band beyond a multiple of its child tile size (opening-criterion analogue); links to hole 3.
- Gravity needs ~log2(domain/tile) tiers, top tier single tile; LJ needs tier 0 only.
- Not fixed: tight clumps just outside the band at second order (falls to tier 0); channel count grows with order/dim; velocity-dependent coarse effects need momentum moments.

(Hole 2: user moved on without changes.)

### Hole 3 (proposed, pending user response)
- Tier 0 trained on scenes no larger than its reach (tile + halo): truth = exact near field, no pair decomposition needed, no mean-field bias (also addresses hole 4).
- Taper mass contributions to zero over outer halo (edge weight exists in scatter_field.py) to avoid force jump at halo boundary.
- Tier k+1 trained on larger scenes against residual (truth - frozen lower-tier output); overlap with lower tiers is absorbed by residual learning. Repeat up the stack.
- Reach constraint: tier k reach must cover tier k+1 near band (from hole 2), else a gap; check 32 tile + 16 halo against opening distance.
- Not fixed: tier 0 near-field errors pass into tier 1 residual; implicit (not explicit) double-count avoidance, explicit per-target exclusion is hard in conv grid form; data cost multiplies per tier; walls/boundaries break "nothing beyond halo" scenes.

### Hole 3 amendments (user input 2026-10-24)
- Tier 0 must see inputs shaped by dynamics it cannot see (bulk acceleration, tidal shear, hidden state, entering particles) but target must stay near-field only (else mean-field bias, hole 4).
- Proposed: impose random smooth external field (uniform + gradient, per scene) on tier-0 training scenes; state evolves under near + external; target = truth - external (known, no pair decomposition). Mirrors tier k+1 local-expansion output; at deployment tier 1 replaces it.
- Error-estimate channel per cell passed to next tier (user idea). Value: exports sub-tile reliability lost in coarsening; learned opening-criterion analogue; depth/refinement gate (hole 8). Limits: OOD calibration (least reliable where errors largest); much of tier-k error is the far field tier k+1 supplies anyway; tier k+1 can only gate/weight, not repair, fine-scale errors; aggregation rule (sum/mean/max) is a choice; one-step force error is a cheap target but drift is not per-cell attributable; downstream use must obey hole-1 constraints.
- Alternative needing no extra training: physics proxy (grid continuity residual, residual-channel curl). Test arms: learned channel / proxy / neither.

### Hole 4 (proposed, pending user response)
- Mostly subsumed by hole 3 fix (near-field-only target; external field subtracted).
- Residual issues: (1) synthetic external field (uniform + tidal) narrower than real far field (higher order, time-varying); (2) real far field correlates with local density, independent sampling hides this -> input-distribution mismatch at deployment; (3) bias needs a check.
- Check: tier 0 output on a window cut from a large rollout vs isolated cutout near-field truth; mean signed difference = bias. Pass: consistent with zero within seed noise. Kill: systematic offset correlated with local input.
- Fallbacks if hole 3 approach rejected: (a) large scenes + oracle far-field input channel (train/test mismatch when swapped for tier 1); (b) large scenes, full truth, no conditioning (original biased form).

### Hole 5 (proposed, pending user response)
- Three pieces: exposure bias (tier 1 never sees tier 0 rollout-error states), coupling when tier 0 is in the loop, no adaptation of tier 0.
- Step 1 diagnostic: tier 1 residual error on teacher-forced states vs tier-0-rollout states; gap within seed noise -> teacher forcing suffices.
- If gap: (a) rollout-in-loop fine-tune of tier 1 with tier 0 frozen (grads through tier 0 states, update tier 1 only; truncated backprop if memory-bound); (b) noise injection matched to tier 0 measured rollout error (cheaper, keeps tier 1 less tied to one tier 0).
- Same-state design: tier 0 does not depend on tier 1, nothing to adapt. Optional joint fine-tune arm must keep tier 0 near-field loss on small scenes or hole 4 returns.
- Not fixed: tier 1 specific to the tier 0 it trained with (retrain on swap); horizon choice; gradient memory.
- Arms (each vs frozen baseline): teacher-forced, rollout-in-loop, noise injection, joint fine-tune.

- User decision (hole 5): a tier 0 trained for a specific tier 1 is acceptable; tiers are parts of one larger model. Cost (tier 1 specific to its tier 0; retrain on swap) is kept but at slightly lowered priority, still a consideration. Possible later: fine-tune pretrained tiers from different runs to work together; requires compatible interfaces (channel definitions, moment order, grid resolution, tile size at each boundary).

### Hole 6 (proposed, pending user response)
- Same G x G grid at every tier, tile spans 2x linear extent; each parent cell = 2x2 merge of child cells; 4 child grids -> 1 parent grid (4x fewer tiles per tier in 2D, 8x in 3D); total cost geometric (~4/3 of tier 0 in 2D).
- Merge exact (parallel-axis for moments). Add velocity dispersion tensor (merges exactly via law of total variance); candidate position-velocity cross moment (rotation/shear). Minimal channel set chosen by test.
- Optional learned latent channels per cell with small learned merge (gives up exact-merge guarantee).
- Normalise moments by cell size (mass per area); gravity is scale-free so tier k+1 could be initialised from tier k (test).
- Halo at each tier's own scale; compute factor = halo ratio, sum stays O(N).
- Not fixed: info beyond chosen order/channels lost; empty cells still cost (hole 8).
- Arms: exact moments only / + velocity dispersion / + learned latent.

### Recurrent state placement (user question, 2026-10-24; recommendation)
- Default: per-tier, per-cell hidden field on that tier's grid (existing scatter_field mechanism: previous output field as input + hidden_ch); tile holds a patch; weights shared within tier.
- Per-particle hidden vector at tier 0 only, for particle-intrinsic history (Lagrangian; moves with particle across tile boundaries; scatter/gather as channels).
- One vector per tile rejected (too coarse).
- Cross-tier: tier k hidden merged upward via learned operator (hole 6 latent channels) into tier k+1; nothing downward needed.
- Halo state must be refreshed from neighbour interiors each step or seams result (hole 7).
- Upper tiers default off for recurrence (Newtonian particle state is Markov); test as an arm.
- Full rollout/checkpoint state = particles + every tier hidden field + per-particle hidden.

### Parent output as child's next-step input (user idea, 2026-10-24)
- One-step-lagged top-down conditioning (FMM downward pass in time); removes within-step ordering problem; lag accurate because far field varies slowly.
- Extends hole 3: imposed external field becomes an explicit input channel (value, gradient, Hessian) to tier 0 during training; at deployment filled by parent's lagged local-expansion output. Keeps tier-0-first order.
- Physical outputs (potential/grad/Hessian) have synthetic stand-ins -> compatible. Recurrent latent states have no stand-in -> would need joint/reversed training; deferred.
- Risks: feedback loop tier0<->tier1 across time (stability, BPTT through loop); train/test mismatch (clean imposed vs erroneous parent; noise injection as in hole 5); over-reliance on channel (limited by near-field-only target); hole 1 curl-free guarantee unaffected (per-tier potentials still sum).
- Test arm: tier 0 with vs without channel on strong-external-field scenes; keep if near-field error gain > seed noise and rollout energy drift not worse.

### Hole 7 (proposed, pending user response)
- Single ownership: each fine cell/particle takes output only from the parent whose interior contains it; halo outputs discarded (as tier 0 does). Input-side halo cells are read-only copies of merged moments, never re-summed.
- Seams are a conservation problem: potential jump at parent boundary = energy kick on crossing particle (same class as LJ non-conservative injection).
- Blend POTENTIALS in a band inside the overlap with partition-of-unity weights (bilinear/cosine), then take gradient of blended function including grad-w term; do not blend forces (breaks curl-free, hole 1).
- Seam-consistency loss: penalise adjacent parents' output difference in overlap (local, label-free).
- Keep hole-3 input taper and this output blend as separate mechanisms.
- Particle handoff: per-particle hidden moves with particle; Eulerian hidden halo refreshed each step.
- Validation: existing seam-video comparison (tiled vs single vs exact) per tier + potential-jump measurement at each seam.
- Not fixed: halo >= blend band (cost); blend zones evaluate 2 parents (4 at 2D corners, 8 in 3D); different tier timescales -> stale halos (hole 10).

### Hole 8 (proposed, pending user response)
- A. Sparse tile activation: tier-0 tile active if it holds particles or lies within one step's max displacement of an active tile; missing halo neighbours = zero; upper-tier parent active only if a child is. Cost ~ active tiles (O(N), bounded by domain tile count). Dynamic batching; static-shape CUDA graphs need capacity buckets.
- B. Dense-core refinement = extra RESIDUAL tier below tier 0 (2x2 subdivision, same G x G grid, half cell size), active only where dense; trained on dense scenes against frozen tier 0 residual (avoids double counting with tier 0's full near-field target). Tier index no longer strictly scale order; training chain unchanged.
- Coarse-fine consistency: exact moment merge (hole 6); potential blended across refinement boundary (hole 7).
- Smooth gate g(rho) in [0,1], not hard switch; differentiate gated potential g*phi (adds phi*grad-g term) to stay curl-free.
- Refinement criterion: local density baseline; error-estimate channel as test arm.
- Aligns with project note: adaptive FMM with learned kernel favoured (not started).
- Not fixed: LJ fixed length scale -> refined level trained separately (gravity may share weights); activation scheduling/hysteresis; more training scenes per refinement level.

(Hole 8: user LGTM.)

### Hole 9 (proposed, pending user response)
- Shared with U-Net: per-level weights (per-scale specialisation) and up/down scale info flow already exist in a U-Net.
- Added by hierarchy: unbounded range at bounded cost per tile (log depth); sparsity/adaptivity (hole 8); exact moment aggregation vs learned pooling; modular training + per-tier interpretability.
- Lost vs U-Net: end-to-end joint gradient (tiers sequential/frozen); same-step cross-scale interaction in one pass.
- Honest baseline is not a deeper single U-Net (cannot scale) but the existing tiled U-Net + ghosts + far-field path. Function names in scatter_field.py (kernel_acc, _kernel_fft, far_acc) and notes about the learned far kernel suggest an FFT-based global far path; UNVERIFIED, functions not read. If so, hierarchy claim = tiers >=1 replace global far kernel with local sparse learned hierarchy; tier 0 = existing scatter field.
- Comparison, matched compute/domain: A deeper single-tile U-Net on larger grid; B current tiled + ghosts + far path; C tiered. Metrics: far-field force error vs exact, near-field regression, energy drift, cost scaling with N and empty fraction.
- Pass: C beats B on far-field accuracy at matched cost without near-field regression, or matches B accuracy while gaining sparsity. Kill: no gain over B on either axis.

### Deferred: adjacent-tile communication (user, 2026-10-24)
- Current mechanism: halo overlap (+ halo state refresh for recurrent fields). One candidate: U-Net-style skip connection between neighbouring tiles. Other options to be explored by a research agent; dispatch later and only if needed (not dispatched).

### Hole 10 (proposed, pending user response)
- A. Timescales: synchronous (all tiers every step) by default. Upper tiers add ~1/3 of tier-0 cost (hole 6 geometric sum) = at most ~1/4 of total before halo overhead; project notes record adaptive/substep routing negative earlier.
- Multi-rate test arm: scatter_field.py has _step_respa (reversible multi-rate). Slow half-kick (upper tiers) around m fast tier-0 steps; symplectic, no stale-state bias; stale halos moot if all tiles of a tier update together.
- RESPA needs potential-derived forces (hole 1); residual non-conservative channel breaks it -> synchronous for those systems. Side effects: parent->child lag becomes m steps; tile activation refreshed at outer steps with displacement bound scaled by m.
- B. Domain edges: all symmetry-breaking enters as data channels, never different weights. Boundary occupancy/SDF + type mask as input (as scatter-contact option C did); open = zero-mass halo (tokens outside grid get zero force per docstring); periodic = wrap-around halo copies; walls = occupancy channel. Upper tiers: coarse occupancy fraction merges by area average. External forces (uniform gravity) added by integrator (consistent with hole 1).
- Not fixed: wall behaviour quality (option C long-horizon runaway) separate from hierarchy; multi-rate multiplier m is an extra hparam.

(Hole 10 part A: user LGTM; part B led to the objects discussion below.)

### Larger irregular objects and collisions (user question, 2026-10-24)
- Representation (recommended): particle cluster, no object-level entity. Shape = arrangement; material properties as per-particle channels; cohesion from same learned short-range field (LJ-solid case). Material channels, not instance ID (identical-material blocks pressed together should act as one continuum); ID is bookkeeping only.
- Object state emerges from moment channels: at the tier where the body ~ one cell, mass, dipole (COM), quadrupole (inertia), momentum, position-velocity cross moment (angular momentum) = rigid-body state. External torque follows from Hessian acting on each particle at its position.
- Alternative test arm for very large objects: rigid SDF body with pose (option C style); needs object-level integrator (assumes rigidity, against learn-the-dynamics) + force/torque gather.
- Collision: approach via tier matching separation (long range) or nothing until tier-0 reach (contact); contact cells show steep gradient + relative velocity; tier 0 outputs repulsive potential gradient; momentum propagates through bodies by cohesion as a wave limited by halo per step; potential channel alone = perfectly elastic, restitution <1 and friction need residual channel / per-particle head with relative-velocity input; off-centre impact gives torque automatically; tunnelling needs per-step displacement < interaction range else substeps.
- Dependencies/gaps: solid cohesion gated on LJ-solid fix (arm showed energy drift; hole 1); hole-8 density gate wrong for objects (uniform solid interior is dense but boring) -> gate on activity (gradients, contact, error estimate); angular momentum not enforced (spurious spin gain/loss for irregular rotating bodies); thin sub-cell features need moments/refinement tier; stiff bodies have high wave speed -> small steps or look soft.

### Training data and evaluation scenes (2026-10-24)
- Tier 0 data: scenes <= tile+halo; softened gravity, LJ gas/liquid/solid, contact balls, bonded irregular bodies, collisions (impact parameter, speed, material, spin), boundary scenes (wall/open/periodic with occupancy channels), edge-crossing particles. Per-scene smooth external field (value/grad/Hessian, maybe time-varying); target = truth - external. Better: harvest initial states from large rollouts and compute far-field expansion at that window by direct sum over outside matter (keeps local-density/far-field correlation; needs pair-decomposable generator, else random fields). Multi-step rollouts; noise-injection variant.
- Tier >=1 data: larger scenes up to span, frozen lower tiers in loop, target = truth - lower-tier output. Truth: direct sum at moderate N, tree/FMM larger; no exact truth at ~1B (conservation + reference path as proxy).
- Refinement tier: dense/contact-rich scenes vs frozen tier 0 residual.
- Eval groups: bias (window cutout vs isolated truth); seams (straddle/cross tile+parent boundaries, potential jump, energy gain on crossing, tiled/single/exact); range (separations spanning tiers, arms A/B/C); gravity systems (Plummer, cold collapse, binary, disc galaxy, BH); conservation (closed LJ solid/liquid from diagnostic, long gravity E/p/L drift); sparsity+refinement (empty fraction, collapsing core, dense/dilute); objects (rest stability, spin, collision sweeps, restitution, fracture, pile on floor, tunnelling); boundaries (open/periodic/wall, box-edge loss from g1); error-channel calibration; exposure bias gap; generalisation (unseen N/domain/mass ratio/material, sub-tile shift invariance); timescale (sync vs multi-rate).
- Metrics: err@k, dE/dp/dL drift, seam jump, cost+memory vs N and empty fraction. Seed 4738 first, held-out seeds for selection. After rollouts: diagnostic grid + unbiased subagent description before reporting.

## Open points (resolved/specified 2026-10-24)
### Tier k>=1 interface (specified)
- Consumes G x G grid (same size as tier 0) with halo copied from neighbouring parents: per cell, exact 2x2 merge of child cells: mass, momentum, spatial moments to 2nd order, velocity dispersion, optional position-velocity cross moment; occupancy fraction for boundaries; optional children's latent summary + error estimate; own hidden field (default off); in feedback arm, lagged local-expansion channel from own parent.
- Emits per-cell local expansion (potential, gradient, Hessian), blended across neighbouring parents (hole 7); optional residual force channel (soft penalty), latent summary + error estimate for next tier, updated hidden field. Force = -grad of summed blended potentials + mean removal. Tier 0 additionally: particle scatter/gather, small per-particle head, per-particle hidden.
- CORRECTION to hole 2: independent gradient/Hessian output heads break curl-free. Define each cell's potential as polynomial phi_c(x) = phi0 + g.d + 0.5 d^T H d (d = offset from cell centre, H symmetric); force -(g + H d) is a true gradient inside the cell. Blend polynomial pieces across cells with partition-of-unity weights (tier-0 gather already bilinear) as well as across parents, else phi jumps at cell edges.
### Sequential-frozen vs joint (test spec)
- Arms vs frozen baseline: S sequential-frozen; S+R S plus rollout-in-loop for tier 1; S+J S plus joint fine-tune of all tiers with tier-0 near-field loss on small scenes retained.
- Pass: improvement over S in far-field error and rollout energy drift beyond held-out-seed noise, hole-4 bias check not regressing. Kill: S+J improves rollout error but fails bias check (tier 0 absorbing far field).


## Implementation status (upper-tier machinery, 2026-10-24, branch worktree-agent-a4059904ca1ecb03c)
Code: `model/tiered/` (opt-in; no tier-0 or generator edits). Tests: `tests/test_tiered.py`, 21 pass (Polaris job 3933, CPU float64).
- `moments.py`: NGP deposit of raw moments about cell centre (m, P, S, Q, W = sum m v v^T, X = sum m d v^T, occupancy); `merge2x2` is exact parallel-axis translation (occupancy = area mean); `moment_features` gives normalised channels (density, mean v, centroid, central 2nd moment, velocity dispersion, optional cross, occupancy).
- `blend.py`: `TileLayout` (GxG tiles, halo copied from neighbours, zero or circular pad). Per-cell polynomial phi0 + g.d + 0.5 d^T H d (H symmetric by parameterisation). C2 smootherstep partition of unity over cell centres and over overlapping tiles (ramp half-width `blend` cells at each tile edge; edge tiles keep weight 1 at an open boundary). One pass returns value, analytic gradient and residual force (2 tiles x 2 cells per axis = 16 pieces per particle). `seam_loss` (overlap coefficient mismatch), `residual_penalty` (magnitude + discrete curl).
- `tier.py`: `TierNet`, weights shared by all tiles of a tier, zero-init head (zero force at init). Outputs 6 poly coefficients (+2 residual), latent + softplus error estimate, hidden field.
- `sparse.py`: `active_tiles` (core mass dilated `dilate` tiles; dilate >= 1 needed so blend neighbours exist), `smooth_gate`, `GatedRefine` (residual tier below tier 0; force = -(g grad phi_r + phi_r grad g), a true gradient).
- `hierarchy.py`: `TieredField(tier0, nets, ...)`. Tier 0 is a black box (`force(pos,vel,mass,mask,state)`, `init_state`, `dt`, `summary_ch`; `ScatterFieldTier0` wraps ScatterField.force/init_field). Tiers >= 1 on merged grids, same-state read, forces summed, mass-weighted mean removed per upper tier, synchronous `step`, `step_respa(m)` (slow half-kicks around m tier-0 steps), optional lagged parent->child (phi, grad) feedback channel, per-tier hidden fields, child latent/error summary (2x2 mean, zeroed in empty cells so activation is invisible).
- Tests cover: merge exactness vs direct sums (2 levels, plus brute-force cell statistics), PU sums to 1 (open/periodic, blend 0/1), blend equals cell-PU reference when tiles agree, force == -autograd(potential), zero circulation across seams, residual channel has curl, potential/force continuity at tile and cell seams, seam loss, translation equivariance (periodic, shift by top tile), momentum conservation (force and steps), sparse == dense, zero-init, hidden/latent/feedback/RESPA smoke, gated-refine gradient.
- Choices to review: NGP (not CIC) deposit for upper-tier moments (offset moments restore sub-cell info; force stays smooth because evaluation is the PU blend); `blend <= halo-2`, `2*blend < G`; residual force channel is blended but is not a gradient (soft-penalised); tier-0 mean removal left to tier 0.
- Not done: training scripts and data, learned-downsampler arm, error-channel proxy arm, refinement criterion beyond density, GPU speed/static-shape batching (only CPU tests run), 3D.

## Experiment: tier>=1 arms S and S+R vs frozen tier 0 + far kernel (2026-10-24)
Setup: frozen tier 0 = Exp E checkpoint (`E_ms_kp_pot_v_g128.pt`: kernel + pp=2 + potential + verlet; weights unchanged, grid widened 128 -> 256 so it covers the 128-unit box at h=0.5). Baseline = tier 0 alone (its learned far kernel included). Scenes: 2-4 Gaussian clumps (40-110 unit-mass bodies each, sigma 3-6, centres >= 24 apart, box +-64), truth = direct-sum fp64 (gs.rollout_torch). Train 160 scenes x 40 steps seed 4738; held-out seeds 9100/9200/9300, 24 scenes x 100 steps. Tiers 1-3 (cell 1/2/4 units, G=16, halo 4, blend 2, pure potential, static shapes), `scripts/train_tiered.py`.
- Arm S: sequential-frozen, one stage per tier (3000 iters, batch 16), loss = |a_up - (a_true - a_tier0)|^2 / |target|^2 on teacher-forced truth states (tier-0 recurrent field carried along truth).
- Arm S+R: S ckpt fine-tuned (1500 iters, batch 8, k 2->8) by unrolling tier 0 (frozen) + tiers in the loop, position/velocity loss.
- Metrics: force_rel (rms |a - a_true| / rms |a_true| over teacher-forced states), com_err (relative error of each clump's mean acceleration = far-field proxy), err@10/50/100, |dE/E| at tick 100.
- PRE-REGISTERED (before any result): PASS if on every held-out seed both force_rel and com_err beat baseline by >10%, and mean err@50 and mean |dE/E| are no worse than baseline mean + baseline seed std; KILL otherwise. S+R judged by the same rule vs baseline (and reported vs S).
- Prediction: S improves com_err markedly (tier 0 far kernel is out of its trained range beyond ~32 units) but force_rel little (tier-0 error already ~5%); rollout err@50 may not improve (exposure bias); S+R recovers part of that.
- Time estimate: S ~15 min, R ~30 min, eval ~5 min, one GPU (smoke job 3937: S ~20 it/s at batch 4, R ~0.3 s/it at batch 4, k 2-3).

### Result (jobs 3937 smoke, 3938 S then S+R; Polaris, ~25 min GPU total; held-out seeds 9100/9200/9300, 24 scenes x 100 steps)
Baseline (tier 0 alone): force_rel 0.055, com_err 0.24/0.30/0.44, err@50 0.74/0.81/0.84, err@100 3.5-3.7, |dE/E| 0.010/0.009/0.012 (truth 2e-4).
- Arm S (3 stages x 3000 iters, ~2 min each; training loss fell only to ~0.6-0.7 of the residual energy): force_rel -12..-14% (improves, seeds 0.049/0.047/0.048), com_err +6..7% WORSE on all seeds, err@50 +5..7% worse, |dE| 1.0-1.4x baseline. Verdict KILL (pre-registered rule: com_err gain must be >10%).
- Arm S+R (1500 iters, batch 8, k 2->8, tier 0 frozen in loop): force_rel -7%, com_err +3..6% worse, err@10 back to baseline (0.0194/0.0205/0.0224), err@50 +1..5% worse, |dE| 2.5-3.5x baseline (0.036/0.030/0.028). Verdict KILL.
- Prediction vs outcome: predicted com_err to improve markedly; it did not. Force_rel improved modestly as predicted. S+R partly repaired short-horizon rollout error, as predicted, but hurt energy drift.
- Reading (untested): the teacher-forced residual (rms ~7% of the force) is mostly per-particle tier-0 near-field error that moment-grid tiers cannot predict; the clump-mean (far-field) part is a small component with this tier-0 (its kernel was apparently adequate to ~128 units here, so the "far kernel out of range" premise did not show up in com_err ordering). The tier nets are local 6-layer convs (reach ~6 cells per tier). No evidence yet on whether the tiers represent far field at all (no oracle far-field-only test run).
- Not run: learned-downsampler arm and error-channel arm (not cheap: new module / per-cell error targets); sweeps avoided on purpose.
- Suggested next diagnostic (not run): train on a far-field-only target (truth minus tier-0 minus direct sum within 20 units, or inter-clump force only) to test whether the tiers can learn the mapping at all, then return to the full residual.

## Experiment: far-field-only target diagnostic (2026-10-24)
Purpose: can the moment-grid tiers represent any far-field signal at all (independent of tier 0)? Target = sum of pair forces from bodies at distance >= 20 units (direct-sum, diagnostic target construction only; tier 0 not used; pair-symmetric so it conserves momentum). Same scenes/seeds/arch as arm S (3 tiers, 6 layers), teacher-forced states, sequential-frozen stages, 3000 iters each. Metric far_rel = rms|pred - target| / rms|target| on held-out seeds 9100/9200/9300 (1.0 = predict zero).
- PRE-REGISTERED: PASS (tiers can represent far field) if far_rel < 0.5 on all three seeds; KILL if mean far_rel >= 0.8; otherwise PARTIAL.
- Conditional follow-ups run in the same job (cheap), judged by the same thresholds: B = reach enlarged (14 layers per tier, 3 tiers); C = 2-tier stack (6 layers). "Reach helps" if B improves far_rel by >20% relative over A on every seed. Full-residual retry of S only after this.
- Prediction: A PARTIAL (far_rel ~0.4-0.7): clump COM forces are smooth and tier-3 cells (4 units) resolve them, but 6-cell reach (24 units at tier 3) is short for 24-60 unit separations; B better than A; C worse than A.

### Result: far-field-only diagnostic (job 3944, ~12 min GPU)
far_rel (seeds 9100/9200/9300): A 3 tiers x 6 layers 0.689/0.611/0.673 (PARTIAL by pre-registered rule: <0.5 all seeds = pass, mean >=0.8 = kill); B 14 layers 0.696/0.622/0.670 (no gain, "reach helps" rule failed); C 2 tiers 0.952/0.911/0.925 (much worse).
- Tiers do represent part of the far field (about 55% of target variance explained) from moment inputs, but training loss plateaus ~0.8-0.9 in sequence (noisy, batch 16). Deeper convs add nothing, so conv reach is not the limiter; the third tier is needed (2 tiers lose most of the signal), suggesting range coverage matters, not per-tier depth.
- Prediction vs outcome: A PARTIAL as predicted; B better than A predicted, wrong; C worse, right.
- Implication: the full-residual KILL is not explained by tiers being unable to learn far field at all, but by far-field being a small share of the residual plus capacity/optimisation limits (single-stage, 3000 iters). Retry of S on the full residual not run (coordinator gating: only after reach test, which was negative). Untested next levers: more training scenes/iterations, a third-order check on moments, tier-3 top-cell resolution.

## Experiment: far-field levers vs arm A (2026-10-24)
One lever at a time, same far-field target, same held-out seeds and metric (far_rel). PRE-REGISTERED: PASS if far_rel < 0.5 on all three seeds; STALL if mean improvement over arm A (mean 0.658) is < 10% relative; otherwise PARTIAL-IMPROVED. Order: (1) D = 10x scenes (1600) + 10x iterations (30000/stage), resume-capable; (2) E = finer top-tier cells / higher-order moments; (3) F = dilated convs only if (1) STALLs. If any arm passes, retry full-residual S with that config vs Exp E baseline under the original rule (force_rel and com_err >10% better on every seed, err@50 and dE no worse than baseline mean + std).
- Prediction (1): undertraining is part of it; D far_rel ~0.45-0.55, PARTIAL-IMPROVED or marginal PASS.
- Estimate (1): scene gen ~10 min, 3 stages x ~11 min.
- Lever (2) implemented as E = third-order moments (4 extra channels: sum m d^3, exact parallel-axis merge, central third moments as features; `--order3`), everything else = arm A (160 scenes, 3000 iters, 6 layers). Tests: 23 pass (job 3952). "Finer top cells" not run separately (2-tier result C already shows coverage loss; G=32 variants change reach). Queued after D (dependency) so only one GPU job runs.
- Prediction (2): E ~0.60-0.65 (small gain; sub-clump shape matters little for forces at >=20 units from 3-6 sigma clumps).

### Status note (2026-10-24, later)
Lever D (job 3950) stayed PENDING (QOSMaxGRESPerUser, both account GPUs held by other jobs) for hours; E (job 3953, afterany:3950) queued behind it. Polaris SSH then dropped (session lost; polaris_start returns "Error reading SSH protocol banner" on repeated tries). No D/E results yet. Resume: polaris_start (Duo), check 3950/3953, read ~/augur-tiered-farD-3950.log / farE, run `--mode far` results in results/far_D.json, far_E.json; then dilated-conv lever if D stalls, else full-residual S retry per pre-registration above.

### Result: far-field levers (jobs 3950 D 40 min, 3953 E 4 min; Polaris GPU)
far_rel on seeds 9100/9200/9300 (arm A 0.689/0.611/0.673, mean 0.658):
- D (10x scenes = 1600, 10x iters = 30000/stage): 0.574/0.518/0.549, mean 0.547, 17% better than A on every seed. Pre-registered verdict PARTIAL-IMPROVED (not pass: needs <0.5 on all seeds; not stall: >10% gain). Undertraining is real but the gain is modest for 10x data and compute; stage-3 train loss ~0.25 (squared) vs held-out rms 0.55 suggests a gap that is not purely generalisation noise (single-batch loss, noisy).
- E (third-order moments, A budget): 0.689/0.622/0.678, mean 0.663 vs A 0.658: STALL (no gain). Sub-cell shape is not the limiter; the exact 3rd-order merge is tested and kept as an option.
- F (dilated convs) not run: its trigger (D stalls) was not met. S full-residual retry not run: trigger (a lever passes) not met.
- Prediction vs outcome: D predicted 0.45-0.55 (landed at the top edge, 0.55, no pass); E predicted 0.60-0.65 (landed 0.66, stall).
- Reading: input moment order and conv depth/reach are not limiting; more data/iterations helps slowly. Open (untested) levers: 100x budget along D's trend, tier-3 top cell/tile layout changes (G, halo), loss weighting toward large-force particles, joint (not sequential-frozen) training of the three tiers.

## Experiment 1: full-residual S retry with D config (2026-10-24)
Config = original arm S (residual target vs frozen Exp E tier 0, 3 tiers, 6 layers) with D's budget: 1600 scenes x 40 steps, 30000 iters/stage, batch 16. PRE-REGISTERED, same rule as the original S: PASS if on every held-out seed (9100/9200/9300) force_rel and com_err both beat baseline by >10% and mean err@50 and mean dE are no worse than baseline mean + baseline seed std; KILL otherwise. Original S for reference: force_rel -12..14%, com_err +6% worse, err@50 +5..7%.
- Prediction: force_rel gain grows to ~15-20%; com_err stays within +-5% of baseline (far-field is a small share of the residual; D showed far-only far_rel 0.55); verdict KILL.
- Estimate: targets ~5 min, 3 stages ~40 min, eval ~5 min.

### Result: Experiment 1 (job 3963, ~40 min GPU, results/tiered_eval_SD.json on Polaris)
S with D budget: force_rel gain +28.7/+28.9/+28.1% over tier-0 baseline (0.056 -> 0.040), com_err 0.0/-5.0/-1.3% (no gain; baseline 0.24/0.30/0.44), err@50 +4%/+3%/-1% vs baseline (0.771/0.836/0.829 vs 0.740/0.808/0.841), err@100 within +-3%, dE_model ~1.0-1.7x baseline (0.0129/0.0141/0.0136). Pre-registered verdict KILL (com_err gain <10%). Prediction partly wrong: force_rel gain was 28%, not 15-20%; com_err and verdict as predicted.
- Reading: upper tiers now remove a third of the force error but the clump-scale acceleration error (com_err) is untouched, so it is not far-field-limited by capacity/data. com_err probably comes from tier-0's near-field window bias (clump-edge truncation) that the frozen tier-0 cannot be corrected for by a smooth upper tier. This motivates Experiment 2 (joint fine-tune with tier 0 trainable).

## Experiment 2: S+J joint fine-tune of tier 0 + all upper tiers (2026-10-24)
Init: Experiment 1 S nets (tiered_D/tiered_S.pt) + Exp E tier 0. Train (mode J) all tiers (lr 1e-4) and tier 0 (lr 1e-5) jointly on free rollouts k 2->8 of 400 multi-clump 128-box scenes (loss: position + 0.1 velocity MSE), PLUS tier-0 near-field retention loss (tier 0 alone, k-step rollout on 400 small 10-100-body scenes, weight 1.0). 4000 iters, batch 16, seed 4738, cosine lr. Eval: same held-out seeds/metrics as Exp 1; bias check (mode bias) = tier 0 alone vs isolated-clump truth inside an 18-unit window: near_rel (rms err / rms target) and slope (projection on far-field force).
- PRE-REGISTERED: PASS if on every seed force_rel and com_err beat baseline by >10%, err@50 and dE no worse than baseline mean + baseline seed std, AND bias check holds on every seed (fine-tuned tier-0 near_rel <= 1.10 x baseline near_rel and |slope| <= |baseline slope| + 0.05). KILL if rollout metrics pass/improve (err@50 better than baseline mean) but bias check fails; KILL if com_err gain <10% on any seed regardless.
- Prediction: com_err improves modestly (5-15%), err@50 improves, bias check marginal (near_rel +5-20%). Verdict uncertain; p(PASS)~25%.
- Estimate: 40 min train, 5 min eval/bias.
