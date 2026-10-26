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

### Result: Experiment 2 S+J (job 3976, 53 min train + eval, GPU)
Held-out 9100/9200/9300 vs tier-0 baseline: force_rel +28.2/+26.9/+25.2%; com_err +25.7/+22.4/+42.6% (0.236->0.176, 0.297->0.230, 0.439->0.252); err@10 -35%; err@50 -12/-13/-13% (0.739->0.649, 0.808->0.704, 0.841->0.732); err@100 -3/-5/-9%. Bias check passes: tier-0 near_rel 0.057/0.053/0.056 -> 0.051/0.046/0.048 (improved), |slope| 0.004/0.001/0.002 -> 0.008/0.005/0.007 (within +0.05). FAIL: dE_model 0.067/0.061/0.063 vs baseline 0.010/0.009/0.012 (6-7x; true 0.0002), baseline mean+std = 0.0129 -> rollout_ok False. Pre-registered verdict: KILL on the energy criterion (not on the bias criterion). Prediction was off: com_err gain far larger than 5-15%, bias fine.
- Reading: first config that clears com_err and err@50; the only failing criterion is energy drift, which the loss (position/velocity MSE over k<=8 steps, no energy term) does not constrain; upper tiers were pure-potential here (--residual off), so drift likely comes from the momentum-preserving mean removal (mean_remove=True, non-conservative) and tier-0 co-adaptation; untested. Not a tier-0 near-field regression (near_rel improved). Next lever (not run, outside the 3-item list): add the energy-similarity term (match truth E(t), per project rule) to J's loss and re-run.

## Experiment 3: structural levers on frozen-tier-0 S (2026-10-24)
Same config as Experiment 1 (1600 scenes, 30000 iters/stage, 3 tiers x 6 layers, residual target vs frozen Exp E tier 0), one structural change per arm, sequential in one job: Z = 4 latent summary channels passed up with fixed 2x2 average pooling (control); L = Z with a per-tier learned 2x2 stride-2 conv downsampler; Er = Z plus a predicted-error channel per tier (supervised by cell-mean of |remaining residual|/rms, weight 0.1, cumulative-detached) used as a proxy summary passed up. Top-tier tile/halo layout NOT run (needs layout-code changes; earlier reach tests B (depth) and C (2 tiers) gave no gain/worse).
- PRE-REGISTERED: each arm judged by the original rule (force_rel and com_err >10% better than baseline on every seed; err@50, dE no worse than baseline mean + std) -> PASS/KILL; lever counts as HELPING if its mean force_rel gain or com_err gain exceeds Z's by >5 points absolute. Exp 1 reference: force_rel +28.7/28.9/28.1%, com_err 0/-5/-1%.
- Prediction: all three KILL on com_err (frozen tier 0 bias is the limit, as Exp 2 suggests); L and Er within 3 points of Z. Est. 40 min per arm.

### Result: Experiment 3 (job 3979), arms Z and L (Er pending at time of writing)
Z (4 latent channels, fixed pooling): force_rel +28.5/29.2/28.1%, com_err -1.4/-3.6/-0.5% -> KILL. L (learned 2x2 downsampler): force_rel +28.5/28.7/28.1%, com_err +3.1/-6.3/+2.5% -> KILL. Neither beats Exp 1 (S, no latent): latent summaries and learned downsampling do not help; consistent with prediction (frozen tier-0 bias is the limit).

## Experiment 4: energy term and drift ablations on S+J (2026-10-24)
Base = Experiment 2 (J: init Exp 1 S nets, 4000 iters, rollout k 2->8, near-field retention). Arms, one change each vs J (J dE_model 0.067/0.061/0.063 vs baseline 0.010/0.009/0.012):
- JE: add energy-similarity loss, mean over rollout steps of ((E_model(t) - E_true(t))/|E_true(t)|)^2 with E_true from the truth state at the same step (never E(0); works for non-conserving truth), weight 1.0.
- JM: no mean removal on upper-tier accelerations (diagnostic only; breaks momentum conservation by construction).
- JT: tier 0 frozen (lr0 = 0), upper tiers still trained jointly on rollouts.
- PRE-REGISTERED: JE judged by the full Exp 2 rule (force_rel and com_err >10% better than baseline on every seed, err@50 and dE no worse than baseline mean + std, bias check as Exp 2) -> PASS/KILL. Drift attribution: factor X (mean removal / tier-0 co-adaptation) is the cause if its ablation cuts mean dE_model below half of J's (<0.031); dominant if below 2x baseline mean (<0.020); otherwise not the cause. JM/JT also report rollout metrics but are not candidates for adoption.
- Prediction: JE passes dE (energy loss directly penalizes drift) but may lose 5 points on com_err; JT dE stays high (the upper tiers drift on their own); JM dE drops moderately. p(JE PASS) ~45%. Est. 55 min per arm, ~3 h total.

### Result: Experiment 3 complete (job 3979, ~2.2 h GPU)
Er (latent + predicted-error channel): force_rel +28.8/29.2/28.3%, com_err +2.2/-0.2/-0.5%, err@50 +4%/+1%/+2%, dE ok -> KILL (com_err). All three structural arms (Z, L, Er) are indistinguishable from Exp 1 S (force_rel ~+28.5%, com_err ~0): none helps, so summary-channel design and learned downsampling are not limiting; the frozen tier-0 bias is. Tile/halo layout lever not run.

### Result: Experiment 4 (job 3985, private dir ~/augur_tiered_c2, 3 arms ~55 min each)
Mean dE_model over seeds (baseline 0.0104, Exp 2 J 0.0625; threshold "cause" <0.031, "dominant" <0.020):
- JE (energy-similarity term, w 1.0): dE 0.051/0.044/0.048 (mean 0.048, -23% vs J); force_rel +27.5/26.3/24.5%, com_err +26.3/22.9/43.7%, err@50 -14..-15%; bias OK (near_rel 0.051/0.046/0.048). Verdict KILL on dE (still 4.6x baseline). Energy term helps only partially: loss covers k<=8 steps, drift is measured at tick 100.
- JT (tier 0 frozen, upper tiers rollout-trained): dE 0.035/0.034/0.031 (mean 0.033, -47% vs J, just above "cause" line); force_rel +15..17%, com_err -2..-4% (no gain). KILL. Drift persists without tier-0 co-adaptation, so rollout-trained upper tiers are the main drift source (S alone: 0.013-0.016); tier-0 co-adaptation adds the rest but is also what delivers com_err.
- JM (no mean removal): dE 0.065/0.055/0.062 (mean 0.061, -3% vs J); com_err gain smaller (+13..37%). Mean removal is NOT the cause.
- Attribution: mean removal no; tier-0 co-adaptation partial (about half); rollout training of the upper tiers the rest. Untested hypothesis: tier potentials are coarse approximations of pairwise potential so -grad(coarse potential) does not conserve the true pairwise energy under long rollouts; energy loss over longer k or an explicit energy-drift penalty at long horizon is the next candidate.

### Checkpoint locations (Polaris, /home/adamwebb; verified by ls 2026-10-24; correction to D2's note that Z/L/Er saved none)
- Exp 1 S: augur-tiered/checkpoints/tiered_D/tiered_S.pt (targets_4738_1600_40.pt alongside). Exp 3: tiered_S_Z.pt, tiered_S_L.pt, tiered_S_Er.pt in the same dir; load with --tiers 3 --latent 4 plus --learned-pool (L) or --err (Er), --tag _Z/_L/_Er.
- Exp 2 J: augur-tiered2/ckpt_J/tiered_J.pt (nets + tier0 + opt, 5 MB). Exp 4: augur_tiered_c2/ck_JE|JT|JM/tiered_J.pt; results augur_tiered_c2/res_{JE,JT,JM}_{eval,bias}.json. Eval JSONs for Exp 2: augur-tiered2/results_J_{eval,bias}.json.
- D2 note (sparse tile activation 4.7x at 1024-unit span, 12.5% occupancy; RESPA m=4 passes on J, err 1.06x, cost 0.54x) recorded as reported by coordinator, not verified by me. D2 also reports J tier-0 bias still fails (max|z| 4.7) under its own test, which conflicts with my bias check (near_rel/slope pass); different test, unreconciled.

## Experiment 5: energy-form tiers (2026-10-24, coordinator request)
Architecture. The polynomial-potential tier (control) has phi(x; moments(t)): a gradient in x but time-dependent, so no single energy functional of the particle positions exists and energy is not conserved along rollouts (Exp 2 dE 6x baseline; Exp 4 JE only -23% with an energy loss). Energy-form tier (model/tiered/energy.py, TierNetE/TieredFieldE): each tier k computes a smooth (CIC) mass density on its grid, a conv net on padded tiles outputs a per-cell energy density, U_k = sum of interior cells (position-only, memoryless, no hidden/latent state, no velocity input, no mean removal), and the acceleration of particle i is -dU_k/dx_i / m_i by autograd through the CIC deposit (create_graph for training). Same idea as Agent A's LJ mesh fix (branch hole1-potential-field). Tests: finite-difference check of force vs U, and zero work around a closed path (both pass, CPU float64). Trade-off: input is the density only (no velocity dispersion or higher moments), so its fit may be weaker than the polynomial-potential tier.
- Arms (everything else as the polynomial control, same seeds/scenes/tiers/layers/width): EN-S = sequential-frozen S vs frozen Exp E tier 0 (as Exp 1); EN-J = joint fine-tune of EN-S nets + tier 0 (as Exp 2: 4000 iters, k 2->8, near-field retention, lr 1e-4/1e-5, no energy loss). Controls (already run): Exp 1 S, Exp 2 J.
- PRE-REGISTERED: EN-J PASS if on every seed force_rel and com_err >10% better than tier-0 baseline, err@50 and dE_model no worse than baseline mean + baseline seed std, and bias check as Exp 2 (near_rel <= 1.10x baseline, |slope| <= baseline + 0.05). Mechanism criterion (reported separately): dE_model <= 2x baseline mean (0.021) for EN-J, vs 6x for J. KILL otherwise. EN-S judged by the Exp 1 rule (reported for the force fit).
- Prediction: dE drops sharply (to <=2x baseline, p 70%) because only tier-0 residual and integrator error remain; force_rel gain smaller than polynomial form (about +15..22% vs +28%), com_err gain 10-30%; p(full PASS) ~30%.
- Cost: a smoke job first measures s/iter (double backward through CIC); iteration budget for EN-S is set from it and stated in the result (a smaller budget than Exp 1's 30000 would be a confound and will be flagged).

### Result: Experiment 5 arm EN-rho (density-only energy tier; job 3999, cancelled after stage 1, 29 min)
Smoke (job 3998) passed end to end (26 tests, S/J/eval/bias run). Speed 0.057 s/iter at batch 16 (30000 iters/stage ~ 28 min). Stage-1 training loss (normalised squared residual error) plateaued at about 0.9 (0.78-0.97 over 30000 iters) vs 0.57 at 5000 iters for the polynomial tier (Exp 1, same targets, same budget). The density-only energy tier barely fits the residual (<~10% variance explained at tier 1) -> aborted as a fit KILL (not run through eval; later stages could only add far-field). Reading: with density alone the tier has no sub-cell structure at h=1 (tier-0 residual is mostly near-field) and the force is the derivative of a learned scalar, which is harder to fit than a directly predicted potential.

### Experiment 5b: EN-M (energy tier with CIC moment inputs), pre-registered before running
Same energy form; the tier net also receives the CIC-weighted first and second offset moments about each cell centre (5 extra channels, continuous in positions, position-only), the analogue of the polynomial tier's centroid/quadrupole inputs. Everything else as Exp 5. Rules and passes identical to Exp 5 (EN-J full rule; dE mechanism criterion <=2x baseline mean = 0.021). Added kill-early rule: if stage-1 train loss at 10000 iters is >0.8 (polynomial control: ~0.57 at 5000) stop and log KILL (fit). Prediction: stage-1 loss ~0.6-0.7; if the fit is adequate dE drops to ~1.5-3x baseline; p(full PASS) ~25%.

## Experiment 6: long-horizon energy penalty on S+J (pre-registered before running; runs after Exp 5b)
Polynomial-tier S+J exactly as Exp 2 (init Exp 1 S nets) with the energy-similarity term of Exp 4 (weight 1.0, per-step ((E_model - E_true(t))/|E_true(t)|)^2 against the truth state at the same step; never E(0)), but the free-rollout horizon is raised from k 2->8 to k 2->24 (eval of dE_model is at tick 100). 3000 iters (instead of 4000, for cost; flagged), batch 16, lr 1e-4 / 1e-5. Arm name JL; checkpoint ck_JL/tiered_J.pt in ~/augur_tiered_c2. Controls: Exp 2 J (k<=8, no energy term; dE 0.0625), Exp 4 JE (k<=8 + energy term; dE 0.048).
- PRE-REGISTERED: PASS on the Exp 2 rule (force_rel and com_err >10% better than baseline on every seed, err@50 and dE no worse than baseline mean + std, bias check as Exp 2). Mechanism: horizon is the missing piece if dE_model(JL) <= 0.5 x dE_model(JE) (<= 0.024); weakly if < JE; not if >= JE. KILL on the Exp 2 rule otherwise.
- Prediction: dE falls to 0.02-0.035 (2-3x baseline) -> still KILL by the rule, mechanism "weakly"; com_err gain stays > 20%; p(PASS) ~15%. Cost ~1.7 h.

## Experiment 7: bias-test reconciliation with D2 (pre-registered; descriptive, no pass/kill)
Question: why do two bias checks disagree on the S+J tier 0 (this branch: near_rel/slope pass; D2: tier-0 max|z| 4.7, still > 2, vs 6.0 for S)? Both are run on the same checkpoints (baseline Exp E tier 0, Exp 2 J, Exp 4 JE/JT/JM, EN-M J when available) with scripts/tiered_bias_reconcile.py. Tests as defined in code: (mine) tier 0 alone on 24 held-out multi-clump scenes (40-110 bodies per clump, box +-64), inside an 18-unit window around the clump-0 centre, vs the isolated-window truth: near_rel = rms force error / rms isolated force, slope = projection of the error on the true far-field force; threshold in the Exp 2 rule: near_rel <= 1.10 x baseline and |slope| <= baseline + 0.05. (D2, tiered_eval.g_bias) tier 0 alone, windows of +-64 units (reach/2) cut around random particles of a 4096-body clumpy scene (256 box, tick 10), 24 windows x seeds 9100/9200/9300; statistic is the mean signed force-vector difference vs the isolated-window truth divided by its standard error across windows (z), pass if |z| < 2 on both axes. Expected: the tests measure different things (magnitude and far-field-direction coupling at a clump vs significance of a small mean vector bias over many heterogeneous windows); D2's z is a significance statistic so can stay large for a tiny bias, and the baseline tier 0 fails it (z 6.0 for S) so it is a property of tier 0 already. Prediction: D2 z for J lower than baseline-S (reproduces ~4.7); my near_rel improved; no contradiction, the "bias fails" statement refers to a pre-existing tier-0 bias that S+J reduces but does not remove. Finding is judged by whether the same ckpts give those two numbers.

### Result: Experiment 5b EN-M (energy tier + CIC moment inputs; job 4002, 2.5 h S + 1.2 h J; ckpts ~/augur_tiered_c2/ck_ENM/tiered_S.pt, ck_ENM/j/tiered_J.pt)
Fit adequate (stage-1 loss 0.48 at 10k iters, 0.65 at 30k; stage-3 0.60; polynomial control ~0.57). EN-M S: force_rel +19/+20/+20% (polynomial S +28%), com_err -13/-8/-5%, dE_model ok (0.011-0.014) -> KILL (com_err). EN-M J: force_rel +25.4/23.9/23.0%, com_err +24.6/27.2/45.3%, err@50 -7/-10/-9%, bias ok (near_rel 0.049/0.044/0.047 vs baseline 0.057/0.054/0.056), BUT dE_model 0.066/0.061/0.064 = identical to polynomial J (0.067/0.061/0.063; baseline 0.010). Verdict KILL (dE; mechanism criterion <=0.021 failed).
- Key finding: making the upper tiers an exact gradient of a position-only scalar did not reduce drift at all. Therefore the S+J drift is not caused by the polynomial potentials being time-dependent. It comes from the joint fine-tuning of tier 0 (JT with tier 0 frozen halved it, 0.033) and from the co-adapted tier-0/upper-tier pair no longer matching a conservative total (tier 0 is a recurrent/learned potential field whose own conservation was only tuned jointly with its original training). Untested next step: EN-M J with tier 0 frozen (expect <= 0.033 and likely near S's 0.013), and an energy term for the tier-0 fine-tune.
- Density-only energy tier (EN-rho) failed the fit; moments are necessary.

### Result: Experiment 7 bias reconciliation (job 4010, same checkpoints through both tests; results ~/augur_tiered_c2/res_bias_reconcile.json)
| tier 0 | hole-4 max abs z over seeds 9100/9200/9300 (x-axis dominates) | hole-4 rel_rms_force_err (3 seeds) | my near_rel (3 seeds) | my slope |
| baseline Exp E | 5.63 / 5.97 / 3.30 (max 5.97) | 3.29 / 1.52 / 1.43 | 0.057/0.053/0.056 | .004/.001/.002 |
| J (Exp 2) | 4.68 / 3.09 / 2.32 (max 4.68) | 3.48 / 1.53 / 1.42 | 0.051/0.046/0.048 | .008/.005/.007 |
| JE | 5.04 / 3.72 / 2.93 | 3.44 / 1.52 / 1.40 | 0.051/0.046/0.048 | |
| JT (tier 0 frozen, control) | identical to baseline (5.63/5.97/3.30) | | identical | |
| JM | 4.64 / 3.03 / 2.24 | 3.49 / 1.53 / 1.42 | 0.051/0.046/0.048 | |
| EN-M J | 4.95 / 3.49 / 2.83 | 3.46 / 1.51 / 1.41 | 0.049/0.044/0.047 | |
Reproduces D2's number (J max|z| 4.7 vs S/baseline 6.0) with identical checkpoints and code: no contradiction in measurement, the two tests measure different things.
- near_rel/slope (mine): tier 0 alone inside an 18-unit window around a clump centre, 128-unit multi-clump boxes; relative rms force error vs the isolated-window truth, and the projection of the error on the true far-field force. It asks "is the near-field force accurate and uncorrelated with the external field": J improves it ~12%.
- hole-4 z (D2): 24 windows of +-64 units cut at random particles of a 4096-body clumpy scene in a 256 box at tick 10 (window edge cuts whole clumps, so the isolated truth is itself off-distribution for tier 0); the statistic is mean signed vector difference / its standard error across windows. It asks "is there a systematic directional bias, detectable given the window-to-window scatter". rel_rms_force_err there is 1.4-3.5 for every model including baseline, i.e. tier 0 fails badly on these windows regardless; the mean bias (-0.006..-0.02 in x) is small in absolute terms and only significant because se is small.
- Both are consistent: J reduces the x-bias (mean -0.0206 -> -0.0201 at 9100, -0.0118 -> -0.0082 at 9200, -0.0088 -> -0.0059 at 9300) by 3-30% but not to |z|<2 for 2 of 3 seeds' x-axis, because the bias exists in the unmodified tier 0 (max|z| 6.0) and S+J only partly moves it. The pass criterion |z|<2 is therefore failed by baseline tier 0 itself; "bias still fails" is a pre-existing tier-0 property, not a regression introduced by tiers (JT = baseline exactly, confirming the upper tiers do not enter this test). Adopting a regression-style rule for tiered arms (|z| not above baseline + 0.5, as tiered_holes.py's own verdict code does) rather than an absolute |z|<2 is the consistent reading; under it J, JE, JM and EN-M J all pass.

### Exp 6 status (job 4010, JL): crashed with CUDA OOM at iteration ~2700 (k=22; memory grows with k, 140 GB). Last checkpoint it 2500 (~/augur_tiered_c2/ck_JL/tiered_J.pt). Training log: large rollout loss grew with horizon (0.0005 at k 8 -> 0.006-0.009 at k 20-22) and the tier-0 near-field retention loss rose 20x (0.000012 -> 0.00025-0.00035) because the near-field rollouts also lengthen with k. Resume with the horizon capped at k_end 20 (schedule differs from the original after it 2500: k jumps from 20 to ~17-20 for the final 500 iters; flagged deviation). Results of the resumed run judged under the Exp 6 pre-registered rule.

## Experiment 8: EN-M with tier 0 frozen (co-adaptation test; pre-registered)
EN-M S nets (ck_ENM/tiered_S.pt) fine-tuned jointly on rollouts exactly as EN-M J (Exp 5b: 4000 iters, k 2->8, near-field loss on tier 0 irrelevant) but lr0 = 0 (tier 0 does not change). Arm ENMT, ckpt ~/augur_tiered_c2/ck_ENMT/tiered_J.pt. Hypothesis from Exp 4/5b: the drift is driven by tier-0 co-adaptation; the energy-form upper tiers cannot drift by themselves. PRE-REGISTERED: supports the hypothesis if mean dE_model <= 2x baseline mean (0.021) vs 0.064 for EN-M J. Full Exp 2 rule also reported (expected FAIL on com_err since tier 0 is frozen: Exp 4 JT com_err ~0).
- Prediction: dE 0.013-0.022 (hypothesis supported, p 65%), com_err gain < 10%, force_rel +15..22%.

## Experiment 9: anchored tier-0 fine-tune (L2-SP) + energy term (pre-registered)
Middle path: polynomial tiers as in Exp 2 (init Exp 1 S), tier 0 trainable at lr0 1e-5 but with an L2-SP anchor to its original weights, penalty 0.05 x (||theta - theta0||^2 / ||theta0||^2), plus the energy-similarity term (weight 1.0, truth E(t)), 4000 iters, k 2->8, near-field retention on. Arm JA, ckpt ~/augur_tiered_c2/ck_JA/tiered_J.pt. Relative weight drift ||theta - theta0|| / ||theta0|| is logged during training (Exp 2/J's drift is computed afterwards from its checkpoint for reference).
- PRE-REGISTERED: PASS if the Exp 2 rule holds (force_rel and com_err >10% better than baseline on every seed; err@50 and dE_model no worse than baseline mean + std (<= 0.013); bias per Exp 2 (near_rel <= 1.10x baseline, |slope| <= baseline + 0.05) and hole-4 max|z| not above baseline max + 0.5). Partial credit reported separately: com_err gain >= 10% on every seed with dE <= 0.5 x J's (0.031).
- Prediction: weight drift well below J's; dE 0.025-0.04, com_err gain 10-30%; p(PASS) ~10%, p(partial) ~45%. A single anchor strength is used on purpose (no sweep); if the partial result appears the strength is the first thing to change and would be separately pre-registered.

### Result: Experiment 6 JL (job 4019 resumed from it 2750 with k_end 20 after the OOM; ckpt ~/augur_tiered_c2/ck_JL/tiered_J.pt)
Vs baseline, seeds 9100/9200/9300: force_rel +18.3/+17.1/+15.4% (down from J's +25..28%), com_err +21.0/+17.9/+41.9%, err@10 -26..-34%, err@50 -14/-15/-18%, err@100 -4/-7/-10%; dE_model 0.0230/0.0204/0.0202 (mean 0.0212; baseline 0.0104, J 0.0625, JE 0.048). Bias: near_rel 0.0589/0.0536/0.0551 vs baseline 0.0574/0.0534/0.0555 (unchanged, within 1.10x), slope 0.0064/0.0043/0.0060 (within +0.05). Pre-registered verdict: KILL (dE 0.0212 > baseline mean + std ~0.0125), but narrowly and with the best trade-off so far. Mechanism: horizon is the missing piece (dE 0.0212 <= 0.024 = 0.5 x JE; 3x lower than J). Prediction (dE 0.02-0.035) correct; com_err gain larger than predicted, force_rel gain smaller (the long horizon trades pointwise force fit for rollout fidelity). Caveats: schedule deviation after the crash (final 750 iters at k 19-20 instead of 22-24; 3000 iters total) and single seed set.

### Result: Experiment 8 ENMT (EN-M with tier 0 frozen; job 4019; ckpt ~/augur_tiered_c2/ck_ENMT/tiered_J.pt)
dE_model 0.0120/0.0113/0.0113 (mean 0.0116 = 1.1x baseline 0.0104; EN-M J 0.064) -> co-adaptation hypothesis SUPPORTED (threshold <=0.021). Energy-form upper tiers on frozen tier 0 add essentially no drift. But force_rel only +10.6/9.3/9.1% (EN-M S was +19..20%: rollout training with frozen tier 0 lost fit), com_err -6.7/-3.4/-2.9% (no gain), err@50 +4/+0.4/+1.8%, err@100 +2%. Bias identical to baseline by construction. Verdict KILL on the full rule; mechanism result is the useful part. Prediction right on dE (0.0116 vs 0.013-0.022 predicted, slightly lower), right on com_err.
- Reading: with a frozen tier 0 the drift is gone but so is the com_err gain; what delivers com_err is tier 0 changing, and that same change is what breaks energy bookkeeping. Middle path (Exp 9, anchored tier 0) and the long-horizon energy loss (Exp 6: dE 0.021 with com_err +18..42%) are the two ways found so far to keep part of the gain.

### Result: Experiment 9 JA (L2-SP 0.05 anchor + energy term; job 4019; ckpt ~/augur_tiered_c2/ck_JA/tiered_J.pt)
Relative tier-0 weight drift ||theta - theta0|| / ||theta0|| (measured from checkpoints; scripts/tiered_t0drift.py): J 0.0163, JE 0.0153, JL 0.0127, EN-M J 0.0170, JA 0.0073 (half of J). JA vs baseline: force_rel +26.6/24.9/23.5%, com_err +25.8/22.5/43.2%, err@50 -15/-14/-14%, err@100 -3..-7%; bias near_rel 0.0516/0.0465/0.0488 (improved vs baseline 0.057/0.053/0.056), slope within tolerance, hole-4 max|z| 5.05/3.89/3.08 (baseline 5.63/5.97/3.30; JL 5.08/4.04/3.14). dE_model 0.0508/0.0449/0.0498 (mean 0.0485 = JE 0.048, J 0.0625, baseline 0.0104). Pre-registered verdict: KILL (dE); partial-credit criterion (com_err >=10%, dE <= 0.031) not met either.
- Reading: anchoring halves the tier-0 weight movement and keeps all of the com_err/err@50/bias gains, but dE does not move beyond what the energy term alone gave (JE). So the energy drift is not proportional to how far tier 0 moves; a 0.7% relative weight change is already enough to destroy tier-0's energy bookkeeping, and the energy term over k<=8 steps cannot repair it. Only the long rollout horizon (JL, dE 0.021) reduced it further. EN-M with frozen tier 0 (ENMT) shows the other bound: dE 0.0116, no com_err gain.
- Summary table (mean over seeds; baseline dE 0.0104):  J 0.0625 / JE 0.048 / JA 0.0485 / EN-M J 0.064 / JM 0.061 / JT 0.033 (frozen tier0, polynomial) / ENMT 0.0116 (frozen tier 0, energy tiers) / JL 0.0212. com_err gain (mean): J +30%, JE +30%, JA +30%, JL +26%, ENMT -4%, JT -3%.
- Pattern: the drift correlates with tier 0 being allowed to move on the rollout loss at all, not with form of the upper tiers, anchor strength or a short-horizon energy term; the horizon (k 20) is the only lever that reduced drift while keeping the gain. Untested next candidates: a longer horizon with a smaller-batch / checkpointed rollout to avoid OOM (k 32+), an energy term on tier 0 directly (E(t) per step in the near-field retention rollouts), and a conservation-by-construction tier 0 (energy-form mesh as Agent A's LJ fix), which is the structural analogue of Exp 5 for tier 0.

## Experiment 10: JL with gradient-checkpointed rollout, horizon k up to 40 (pre-registered)
Same as Exp 6 (polynomial tiers, init Exp 1 S, tier 0 trainable lr0 1e-5, upper tiers lr 1e-4, energy-similarity term weight 1.0 vs truth E(t), near-field retention weight 1.0) but each rollout step (upper+tier-0 field and the tier-0 near-field rollout) is wrapped in torch.utils.checkpoint (non-reentrant, --ckpt-roll) to avoid the OOM at k=22, k ramps 2 -> 40 over 3000 iters, scenes 400 x 80 steps (steps raised from 40 so windows of 40 steps exist). Arm JL40, ckpt ~/augur_tiered_c2/ck_JL40/tiered_J.pt. A second training seed (4739) is run only if the first arm leaves the verdict near the threshold and time allows (cost ~4-5 h each).
- PRE-REGISTERED: PASS on the Exp 2 rule with the Exp 6 dE threshold: dE_model <= baseline mean + baseline seed std (<= 0.0125) on the mean with gains retained (force_rel and com_err >10% better than baseline on every seed, err@50 no worse than baseline mean + std, bias as Exp 2). KILL otherwise. Horizon effect (reported separately): dE(JL40) < dE(JL, k<=20) = 0.0212.
- Prediction: dE 0.012-0.018 (p(dE<=0.0125) ~30%); force_rel gain falls to +10..15%; com_err gain +15..35%; p(PASS) ~20%.

## Experiment 11: gv_NG tier 0 (velocity + recurrent inputs removed) inside S+J, and position-only upper tiers (pre-registered)
Source (Agent A2, branch hole1-potential-field, commit 949a75c, job 4016): on the same 2D gravity benchmark and budget, dE/E@100 for tier 0 variants: gv_R (velocity + recurrent inputs) 0.028/0.023, gv_NG (inputs zeroed: vblind + norec) 0.015/0.014 (= Exp E reference 0.015/0.013), gv_EN (energy-gradient mesh) 0.029/0.028 and 5.6x slower; err@20 equal. So the gradient-mesh variant is skipped (coordinator), and tier 0 = gv_NG (ckpt copied to ~/augur_tiered_c2/gv_NG.pt from ~/augur_slj_hv/ckpt/gv/E_gv_NG.pt; same architecture/weights shape, loaded with vblind=norec=True in scripts/scatter_field.py, widened to 256 grid like Exp E).
Note: gv_R/gv_NG measure the tier-0 alone with its own closed-form metric; Exp E tier 0 already had dE/E 0.015/0.013 so gv_NG is a different checkpoint with about the same drift as the Exp E baseline in this study (baseline dE_model here 0.0104 at 100 steps on multi-clump scenes). The lever to test is whether the S+J-induced tier-0 drift (6x baseline) is smaller when tier 0 has no velocity/recurrent inputs to co-adapt through.
Arms (all polynomial upper tiers, Exp 1/2 recipe: S 1600 scenes x 30000 iters/stage vs the new tier 0, then J exactly as Exp 2: 4000 iters, k 2->8, lr 1e-4/1e-5, near-field retention, no energy term; seeds as before):
- NGa: tier 0 = gv_NG, standard upper tiers (frozen = after S; joint = after J).
- NGb: tier 0 = gv_NG, upper tiers with momentum/velocity inputs removed (--vblind-up: moments built from zero velocity, so mean-velocity and dispersion channels are zero) - tests whether the same lever applies to the upper tiers.
Baseline for each arm = the gv_NG tier 0 alone on the same held-out scenes (so gains are against its own baseline; its dE and bias are reported).
- PRE-REGISTERED: PASS (per arm) on the Exp 2 rule against its own gv_NG baseline: joint force_rel and com_err >10% better on every seed, err@50 and dE_model no worse than baseline mean + std, bias as Exp 2. Lever verdict: "input removal helps" if joint dE_model(NGa) <= 0.5 x dE_model(J, Exp 2 = 0.0625) = 0.031 while keeping com_err gain >10% on all seeds; "helps upper tiers too" if NGb dE < NGa dE by >20% with gains retained; otherwise not.
- Prediction: gv_NG baseline dE about the same as Exp E tier 0 (~0.01); NGa joint dE 0.03-0.05 (inputs removal gives a partial reduction only; p(<=0.031) 30%); NGb no further change (p 70%). Cost ~2.5 h per arm; queued after JL40 (afterany:4021).

### Interim result: Experiment 11 NGa, stage S (job 4024)
gv_NG tier 0 alone (seeds 9100/9200/9300): force_rel 0.0599/0.0582/0.0576, com_err 0.344/0.529/1.009, err@50 0.996/1.065/1.217, dE_model 0.0274/0.0275/0.0504 (this gv_NG checkpoint is not Exp E: 2.6x the Exp E baseline dE here). After S (frozen tier 0): force_rel 0.0372/0.0367/0.0365 (-38% rel), com_err 0.394/0.540/1.015 (worse by 14.5/2.2/0.6%), err@50 -4/-5/-7%, dE 0.0299/0.0344/0.0516 (+9/+25/+2%). Same pattern as Exp 1: sequential fit improves pointwise force but not clump-mean acceleration. S alone KILL on com_err; joint (J) and NGb pending.

### Result: Experiment 11 NGa, stage J (job 4024; ckpt ~/augur_tiered_c2/ck_NGa)
gv_NG tier 0 + polynomial upper tiers, S then J (Exp 2 recipe, no energy term), vs gv_NG alone (seeds 9100/9200/9300): force_rel 0.0372/0.0368/0.0374 vs 0.0599/0.0582/0.0576 (-38/-37/-35%); com_err 0.149/0.182/0.192 vs 0.344/0.529/1.009 (-57/-66/-81%); err@10 -55..-64%; err@50 -35/-34/-40%; err@100 -21/-24/-34%; dE_model 0.0176/0.0121/0.0137 vs 0.0275/0.0289/0.0490 (baseline mean 0.0351, J mean 0.0145); bias near_rel 0.052/0.047/0.049 vs 0.061/0.056/0.057 (ratio 0.84-0.85, better than baseline), slope |<0.0004|.
Verdict (pre-registered Exp 2 rule): PASS on every criterion; lever "input removal helps" PASS (joint dE 0.0145 <= 0.031, com_err gain >10% on all seeds). Prediction was dE 0.03-0.05 (p 30% for <=0.031): beaten by 2-3x. With Exp E tier 0 the same recipe gave dE 0.0625 (6x its baseline); with gv_NG tier 0 (no velocity/recurrent inputs) joint training lowers dE below the gv_NG baseline, so the co-adaptation route (Exp 8) is closed by removing those inputs. Caveats: gv_NG alone drifts 3.3x more than Exp E here (0.0351 vs 0.0106 mean), so the absolute dE of NGa joint (0.0145) is still 1.4x the Exp E baseline; single training seed; pointwise gains similar to Exp E S+J.

## Experiment 12: S1 tier 0 with upper tiers as its external-field supply (design + pre-registration; runs after Exp 10/11 are scored)
Source: coordinator note, E2 published S1 (tier 0 trained from scratch, spec ext=6,extapply,bnd,halo=8,taper=8, third-order imposed far field; plain err@20 1.32x Exp E, tiled(128)/untiled(64) 0.54x; Polaris ~/t0s/results/tier0_ext/S1.pt, repo results/tier0_ext/S1.pt in worktree hetero-tiered-integration, loader scripts/tiered_adapters.load_t0ext). Needs a far-field supply via Tier0.set_external(..., high=(Q, C)).
Interface (first end-to-end hierarchy, imposed-field scheme of the theory doc):
- Domain +-64 (span 128) tiled as in tiered_iface.Tiled: 4x4 tiles of 32, S1 window 64 (tile + 16 halo each side), S1 applied per window to the particles inside it; owned particles take the window's acceleration (this is the same tiling harness as E2/D2).
- Per window w, ext_w = the Taylor coefficients (a0, J, Q, C) about the window centre of the acceleration due to sources OUTSIDE the window. Supplied by the upper tiers: for each window the tier moment pyramid is deposited from the particles outside that window only (a per-window mask, batch dimension B x 16), the tiers produce their blended potential (polynomial tiers, as in Exp 1), and the field is sampled at a 7x7 grid of points inside the window and fitted by a fixed least-squares map to the cubic Taylor coefficients (so the upper tiers never see the window's own sources; tier 0 and the upper tiers have disjoint source sets, no double counting).
- Upper-tier training: stage SE (frozen S1 irrelevant): fit the sampled far field (exact outside-window sources, softened eps 0.5) at the 49 sample points per window, 1600 scenes x 40 steps, 3 tiers x 6 layers, 30000 iters/stage (as Exp 1/D); stage JE: S1 frozen, upper tiers fine-tuned on free rollouts (k 2->8, 4000 iters, energy-similarity term 1.0 vs truth E(t)) with gradient through ext into S1 weights frozen. S1 tier 0 is frozen throughout in this first run (tier-0 co-adaptation drives drift, Exp 8). Arms: OR (oracle ext from exact outside-window sources, no upper tiers: ceiling of S1 tiled), SE (learned ext after stage SE), JE (after rollout fine-tune), and baseline BL = Exp E + far-kernel tiled over span 128 (scripts/scatter_tiling.TiledScatter, 4x4 tiles, far=1, as arm B; checkpoint ~/E_ms_kp_pot_v_g128.pt).
- Metrics on held-out seeds 9100/9200/9300 (24 multi-clump scenes of 40-110 bodies/clump on +-64, 100 steps): err@20, err@50, dE_model, com_err, force_rel (full force vs truth), and far_ext_rel (rms error of the supplied ext field over each window / rms ext).
- PRE-REGISTERED: PASS (hierarchy works end to end) if JE has err@20 and err@50 <= 1.0x BL on every seed, dE_model <= BL mean + BL std, and com_err >=10% better than BL on every seed. Intermediate readings: OR vs BL tells whether S1 tiled can beat the baseline at all (S1's own plain err@20 is 1.32x); SE vs OR tells the cost of learned ext (expected sizeable: far_rel 0.55 in Exp D means 55% error on the far field, but S1 only needs the smooth component); KILL if SE err@20 > 3x OR (learned ext harmful) .
- Prediction: OR err@20 about 1.0-1.5x BL (S1 has 1.32x plain), SE 1.5-3x BL, JE 1.2-2x BL; p(PASS) ~10%. Main value is the first measured end-to-end hierarchy number and ext-error propagation through S1. Cost estimate: implementation + 2 x ~3 h GPU. Queue order after: 4021 (JL40), 4024 (NGa/NGb).

### Result: Experiment 10 JL40 (job 4021, 4.4 h; checkpointed rollout ran to k=40 without OOM; ckpt ~/augur_tiered_c2/ck_JL40/tiered_J.pt)
Vs baseline (seeds 9100/9200/9300): dE_model 0.0150/0.0107/0.0112 (mean 0.0123; baseline 0.0110/0.0090/0.0118, mean 0.0106, std 0.0011; JL k<=20 0.0212, J 0.0625). force_rel +11.8/+10.6/+9.5%; com_err +24.1/+18.5/+42.9%; err@10 -21..-29%; err@50 -15/-16/-19%; err@100 -5/-7/-11%. Bias: near_rel 0.0636/0.0581/0.0594 (baseline 0.0574/0.0534/0.0555; ratios 1.11/1.09/1.07), slope 0.006/0.005/0.006. Tier-0 relative weight drift 0.0149.
Verdict (pre-registered rule): KILL, narrowly, on three near-threshold items: force_rel gain 9.5% on seed 9300 (needs >10%); dE mean 0.0123 passes the numeric 0.0125 written in the pre-registration but fails the code's rule (baseline mean + std = 0.0117); bias near_rel ratio 1.11 on seed 9100 (limit 1.10). Pre-registered prediction dE 0.012-0.018: dE 0.0123 is at the bottom of the range, force_rel +10..15% right, com_err +15..35% right (9300 +43%). Horizon effect confirmed: dE 0.0625 (k<=8) -> 0.0212 (k<=20) -> 0.0123 (k<=40), a monotone dose-response with cost: pointwise force fit falls (force_rel gain +25% -> +15% -> +10%) and tier-0 near-field fit slightly worsens (near_rel +7..11% vs baseline). So horizon trades pointwise fit for energy bookkeeping; at k=40 the trade lands on the threshold of every criterion at once. Second training seed not run (verdict is threshold-limited by a single-seed margin of 0.5 points; running it is the natural next step if this arm matters).
