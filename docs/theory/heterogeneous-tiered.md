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


## Implementation status: tier-0 extensions (2026-10-24, branch worktree-agent-a96914a7f2c210071)
Code: scripts/scatter_tier0.py (module), 8 hook lines in ScatterField._force/force/__init__ (scripts/scatter_field.py, all inert when `model.t0 is None`), `--t0 <spec>` in scripts/train_scatter_field.py, tests scripts/scatter_tier0_tests.py (Polaris job 3936: ALL PASSED). Not implemented by design: potential-gradient output (hole 1 agent), impulse output, speed cap, bounded-neighbour term, mass exchange.
- Spec string flags (`attach_tier0(model, spec)`): moments=1|2, vdisp, cross, head=W, phid=D, phscatter, mat=C, ext=1|3|6, extapply, bnd, halo=H, boundary=tile|open|periodic, taper=T, vscale=S.
- Extra scatter channels: first moment (mean offset), central second moment, velocity dispersion m*sigma^2, position-velocity cross covariance. Raw extensive moments about the cell centre (offsets in units of h) are exposed by `moment_scatter`/`MomentGrid` (15 channels, exact global parallel-axis sums verified to 1e-13 in float64).
- Per-particle head: small MLP (zero-init last layer) on [v, v - local mean flow, |v - u|, mass, log density, mesh accel, ph, material, dispersion]; output acceleration added before momfix (momentum change 3e-7), kept in aux["head_acc"] with `head_penalty` for the soft residual-channel penalty. Per-particle hidden `ph` is carried in the state as `(grid_field, ph)` (works with Verlet and `detach_state`), optionally scattered as channels.
- Per-cell recurrent hidden field (hole 7): already `hidden_ch` + `recurrent`; added halo refresh (`wrap_halo` periodic copy, `zero_halo` open) applied to the field every step.
- Boundaries: `set_boundary(sdf, edge)` -> 5 channels (occupancy, clipped SDF, open/periodic/wall type masks). open = halo particles do not scatter and feel zero force, halo field zeroed. periodic = positions wrapped at core size L = (grid - 2*halo)*h, scatter spill-over folded into the opposite core edge, halo = wrapped copies; force invariant to integer-cell shifts to 1e-7 with a local net whose reach <= halo (a U-Net is not exactly shift-equivariant). periodic is not supported together with kernel/pp terms.
- External field: `set_external(phi0, g, H | grid)` fills [phi], [gx, gy], [Hxx, Hxy, Hyy] channels at cell centres (quadratic potential, force = -(g + H x), a true gradient); `extapply` adds it after momfix so a uniform field is not removed; `grid=` accepts a parent's lagged expansion. Target convention for tier-0 training remains truth minus external.
- Tapered halo: `taper=T` cosine-tapers scattered mass to zero over the outer T cells (receiving force is unchanged).
- Tier-0 interface for tier >= 1: `moment_scatter`, `tier0_forward` (moments, mesh acceleration grid, field, particle accel, state, aux), `model.gather` (bilinear). Merge convention is documented on `MomentGrid`.
- Checkpoint compatibility: new net inputs are appended last, `load_compat` zero-pads them; with a real checkpoint (E_ms_kp_pot_v_g128) flags off is bitwise identical to pre-change on CPU, and with every extra flag on the output is identical (max abs 0) at init. A frozen reference (results/tier0_regress_ref.pt, made from the pre-change file) is checked by the test script, including `t0` attached with all flags off.
- Limits/untested: no training result for any flag (smoke train only: 30 its loss finite); head input scale (vscale) untuned; CUDA graphs and `graphtrain` are bypassed when t0 is attached; mass exchange, adjacent-tile communication and the tile driver itself are not part of this change; periodic wrap with U-Net untested for exactness.

## Arm A: tier-0 with external-field channels (pre-registered 2026-10-24, before launch)
Setup: Exp E net warm-started (load_compat), spec `ext=6,extapply,taper=8,bnd,halo=8`, 6000 windowed gravity scenes (35% uniform-scale, 35% clusters, 30% window-filling gas; 80% with a random smooth external field g ~ 0.003-0.05, H ~ g/16; truth includes it, model applies it exactly so the net learns the near field), 50 epochs = 18750 its at batch 16, k 4->30, lr 5e-4 cosine, seed 4738. Time estimate (job 3940) 0.84 h. Ext channels are fed zero at the hole-4 check (tier 0 alone).
PASS: hole-4 beta (ticks 5/10/20 pooled, held-out seeds, scripts/hetero_tier_checks.py phase a) |beta| <= 0.15 in the 8-16 and 16-32 edge bins on sparse and mid scenes and <= 0.5 on dense (Exp E: sparse -0.04/-0.22, dense +1.3/+1.8), AND plain-scene err@20 <= 1.10x Exp E on eval seeds 9000 and 12000.
KILL: |beta| > 0.5 on sparse or mid, or plain err@20 > 1.25x Exp E. Otherwise inconclusive.

### Arm A result (job 3945, 52 min GPU, 18750 its)
Verdict: KILL by the pre-registered rule (mid |beta| > 0.5); plain-scene criterion passes.
- Plain scenes (Exp E eval seeds), err@5/10/20 arm A vs Exp E: 9000 .0045/.0121/.0379 vs .0045/.0119/.0371 (err@20 1.02x); 12000 .0048/.0128/.0400 vs .0048/.0126/.0395 (1.01x). dE@100 .0166/.0174 vs .0147/.0136. PASS (<=1.10x).
- Hole-4 beta (held-out seeds, ticks 5/10/20 pooled; Exp E baseline in parens): sparse 8-16 -0.24 (-0.04), 16-32 -0.17 (-0.22); mid 8-16 -0.85 (+0.25), 16-32 -1.01 (+0.38), per-seed -0.93..-1.23; dense free-rollout diverges (position error ~1e6 by tick 20, beta meaningless; Exp E dense was finite at +1.3/+1.8). Mid sign flipped and magnitude rose: tier 0 with ext channels fed zero is biased opposite to the far-field direction at mid density, worse than Exp E. FAIL (<=0.15 sparse/mid, <=0.5 dense).
- Evidence the channel is unused: eval on ext scenes with ext channels zeroed gives identical err (.0052/.0139/.0444), i.e. the net ignores the channels; extapply alone carries the field. So the imposed-field training only changed the data distribution (windows with smooth tidal fields, mixed scene kinds), not conditioning.
- Caveats: training scenes are generated independent of the far-field (no local-density/far-field correlation, residual issue (2) of hole 4); ext magnitude (g 0.003-0.05) may be too small to force channel use; hetero check evaluates with grid retargeted to 128 cells over 64 units, taper 8 cells.
- Not diagnosed: why mid went negative, why dense diverges. Candidates (untested): train mix lacks mid density window scenes; the dense divergence is OOD blow-up with no k-gate.

## Arm B1: per-particle head + hidden + material on velocity-dependent (drag) scenes (pre-registered 2026-10-24, before launch)
Why: arm A showed that channels not needed by the training data are ignored; the head must see a force the position-only mesh cannot express. Setup: spec `head=32,phid=8,mat=1,taper=8,bnd,halo=8`, Exp E warm start, same 6000 windowed gravity scenes as arm A but no external field and per-particle linear drag a = grav - gam_i v (half the particles gam=0, others log-uniform 0.02-0.3; mat channel = gam_i/0.3), exact exponential damping in truth; 50 epochs, k 4->30, lr 5e-4, seed 4738. Hidden state phid is expected unused (drag is memoryless); recorded, not scored. Eval on held-out drag scenes (seeds 9000/12000 + 1000): err@20 arm B, arm B head zeroed, Exp E; head fit slope against -gam v.
PASS: drag err@20 arm B <= 0.5x Exp E on both seeds, head-zeroed err@20 >= 2x arm B, AND plain-scene err@20 <= 1.10x Exp E.
KILL: drag err@20 > 0.8x Exp E on either seed, or plain err@20 > 1.25x Exp E. Otherwise inconclusive. Hole-4 bias checks (sparse/mid/dense) reported, not scored. Next (not yet run): B2 LJ scenes, B3 contact scenes.

### Arm B1 result (job 3971, 68 min train, 18750 its)
Verdict: head criterion PASS by a wide margin; plain-scene guard borderline (1.08x / 1.104x vs <=1.10x), so INCONCLUSIVE by the strict rule, no kill trigger hit.
- Drag-scene err@5/10/20 (seed 9000 / 12000): arm B .0050/.0141/.0466 / .0046/.0131/.0421; head zeroed .0117/.0405/.1595 / .0119/.0417/.1611; Exp E .0137/.0446/.1653 / .0150/.0479/.1734. err@20 ratio arm B / Exp E 0.28 / 0.24 (PASS <=0.5); head zeroed / arm B 3.4 / 3.8 (PASS >=2).
- Plain scenes err@20 arm B vs Exp E: .0401 vs .0371 (1.08x), .0436 vs .0395 (1.104x). err@100 2.93/2.87 vs 2.82/2.80; dE@100 .024/.027 vs .015/.013.
- Head fit to -gam*v: slope 1.49 / 1.49, relative residual .50 / .46: it removes most of the error but over-shoots the coefficient and is sloppy; err comes from the rollout loss, not a direct force target.
- The head learned a per-particle velocity- and material-dependent force the mesh cannot express, which arm A's channels did not. Hidden state phid=8 unused-by-construction (drag is memoryless): untested.
- Not run: hetero_tier_checks phase a crashed for head+phid models (per-particle hidden ph size is fixed to the first N while the harness changes N between ticks; harness issue, not fixed). Bias checks for B1 therefore absent.
- Not run (B2 LJ, B3 contact): depend on unmerged branches (lj-scatter-field; scatter-contact) whose LJ failure (mesh injects non-conservative force, open user question on pair-term reciprocity) is unresolved; a tier-0 LJ arm needs that decision first. Raw: results/tier0_ext/armB1_eval.json (on Polaris ~/augur-tier0/results/tier0_ext/).

## Arm A diagnosis, step 1 (job 3980, 2026-10-24)
Measured, no training. Hole-4 mid beta (ticks 5/10/20 pooled, all seeds, 8-16 / 16-32): arm A -0.85/-1.01 (held-out); arm B1 (no ext channels at all, head+drag data) -0.74/-0.84 dev; arm A with taper off at eval -0.88/-1.19; arm A with taper and halo off at eval -0.96/-1.18; Exp E with taper+halo at eval +0.37/+0.60 (untrained flags). B1 sparse -0.22/-0.39; B1 dense diverges like arm A. The hidden-state N-change crash is fixed (ph reset when N differs), so B1 bias ran.
Reading: the mid-density negative bias is shared by two arms with different channels, it is not removed by eval-time geometry flags, so it comes from the trained weights on the common 6000-scene mix (windowed gravity, 50 epochs, warm start), not from the ext channels (arm A) or the head (B1). With `extapply` the imposed field is added exactly and target = truth - ext, so the ext channels carry no information the net needs: unused is expected, not a defect.
Pre-registered next (control A0): same mix, same budget, no ext field (`--no-ext`), spec `halo=8` (no channels). PASS of the hypothesis (bias from mix/fine-tune): A0 mid |beta| > 0.5 at 8-16 and 16-32. REFUTED: A0 mid |beta| <= 0.15 (then the ext path does cause it). Between: inconclusive. Also reruns the channel-weight / gradient / amplitude diagnostic (the first run crashed on a norm call).

### Arm A diagnosis, step 2 (extdiag, job 3981 part 1)
Ext field rms g 0.0127, H*20 0.0132 vs target acceleration rms 1.59 (about 1e-2 of scale, input unnormalised); channel-only gradient norm of loss wrt g 2.9e-7, wrt H 5.1e-6; position change from zeroing channels 3.6e-5; new-channel input weights 0.09-0.97 (some exactly 0) vs 1.64 old mean. Cause: channel amplitude ~100x below the other inputs plus extapply makes the channel redundant. Fix not yet tested: normalise ext channels (scale by target-acc rms / per-scene learned gain) and amplitude curriculum. Control A0 (job 3981, --no-ext, halo=8) produced NaN losses from it ~6300 (non-finite skipped 6500+) and was cancelled: A0 invalid, hypothesis untested. Polaris Duo auth failed on reconnect; stopped to avoid lockout.

## Arms A0 and A1 (pre-registered 2026-10-24, before launch; run in this order, one GPU job)
A0 (control, rerun): `halo=8` only (no channels, no ext field in data), same mix/budget as arm A, lr 2e-4 (was 5e-4, NaN from it ~6300), rollback to last checkpoint after 20 consecutive non-finite steps. Question: is the mid-density negative beta from the training mix/fine-tune rather than the ext path? Reading: mid |beta| >= 0.5 at 8-16 and 16-32 -> bias is from the mix (ext path exonerated, fix must be elsewhere, e.g. mid-density window scenes in data); mid |beta| <= 0.15 -> ext path causes it.
A1: arm A with input gain `extgain=100` (ext channels ~1.3 after gain, matching other inputs) and amplitude curriculum widened to log-uniform 0.003-0.3 (was 0.003-0.05; `--sigma-hi 0.3`), extapply kept, lr 2e-4, same budget. PASS/KILL as arm A: |beta| <= 0.15 at 8-16/16-32 on sparse+mid, <= 0.5 dense, plain err@20 <= 1.10x Exp E; KILL |beta| > 0.5 sparse/mid or err@20 > 1.25x. Additional diagnostic (not scored): zeroed-channel eval difference.

### Results A0 / A1 (job 3986, 50 epochs each, lr 2e-4, no NaN, seed 4738)
Held-out pooled hole-4 beta, bins 8-16 / 16-32 (Exp E in parens: sparse -0.04/-0.22, mid +0.25/+0.38, dense +1.3/+1.8):
- A0 (no channels, no ext field): sparse -0.08/-0.14, mid -0.50/-0.54, dense -0.28/-0.30. Plain err@20 1.00x/0.99x Exp E.
- A1 (extgain 100, amplitude to 0.3, extapply): sparse -0.32/-0.21, mid -0.80/-0.90, dense -0.65/-0.71. Plain err@20 0.995x/0.987x. Zeroing the channels still changes nothing (.0455 both).
Verdict: A0 mid |beta| ~0.5 (borderline KILL line, at the "mix" reading: >=0.5 at both bins, 0.497/0.537), so the mid sign flip and magnitude come from fine-tuning on this scene mix (windowed isolated scenes, 50 epochs), not from the ext path. A1 is KILL (mid |beta| > 0.5); input gain + wider amplitude did not make the net use the channels (extapply makes them redundant by construction) and made bias worse than A0. Dense is finite for A0/A1 at lr 2e-4 (arm A at 5e-4 diverged: the divergence was lr, not ext). Remaining hypothesis (untested): train mix lacks mid-density window scenes / far-field-correlated windows (harvest windows from large rollouts). Ext channels will not be used unless the target needs them (drop extapply with a non-extapply arm, or conditioning on a field the net cannot infer).

### Notes from D2 (tiled use), 2026-10-24
- Arm A (lr 5e-4, ext channels, no gain) multiclump velocity blow-up from step ~35; A0 and A1 (lr 2e-4) stable to step 100. Consistent with job 3986: arm A dense diverged at lr 5e-4, A0/A1 dense finite. Cause is arm A's training run (lr / unnormalised ext path), not a code defect; arm A is KILLed, A1 is the stable ext variant (also KILL on bias). Added aux["max_acc"] (max |a| per force call) for diagnostics; the first-particle log is not implemented.
- B1 tiling crash (hidden size 245 vs 244): `Tier0.begin` now drops the per-particle hidden when N differs from the state (zero reset). A same-N window with different membership would still carry misaligned hidden, so a tiled driver must key hidden state by particle id; not done here.
- Contact balls arm not run (needs ContactScatterField training plumbing).

### Regression after set_ids (job 4003, Polaris CPU)
`scatter_tier0_tests.py --device cpu --ckpt E --skip-train`: ALL PASSED, including flags-off bitwise vs pre-change on the real E checkpoint and all synthetic configs; tests/test_tier0_ids.py passes. The earlier local failure was the local machine's numerics against a Polaris-made reference, not the code.

## Arm H3: harvested windows + Exp E replay + third-order far field (pre-registered 2026-10-24, before launch)
Change from H1 (killed, plain err@20 12.8x): (a) each step's loss is 0.5 x windows + 0.5 x isolated Exp E scenes (1500, 40 ticks, `--replay 0.5`), so isolated accuracy is kept; (b) 120 shards x 32 windows (target ~3800) with the far field expanded to THIRD order (a0, J, Q, C; `scripts/tier0_harvest.py`, expansion about the window centre, sources outside the 64x64 window) and applied exactly by extapply (set_external `high=(Q, C)`); the second-order expansion left 0.30 relative error (H1), the harvest logs error at order 2 and 3. Everything else as H1 (ext=6,extapply,taper=8,bnd,halo=8; Exp E warm start; lr 2e-4; 50 epochs; k 4->20; seed 4738). Hole-4 table now from `scripts/hetero_beta_table.py` (held-out seeds, all three scenes).
PASS: |beta| <= 0.15 at 8-16 and 16-32 on sparse and mid, <= 0.5 dense, plain err@20 <= 1.10x Exp E, multiclump finite with max|v| < 50 to step 100. KILL: plain err@20 > 1.25x Exp E, or |beta| > 0.5 sparse or mid, or multiclump blow-up. Else inconclusive. If the third-order error is still > 0.15 the far-field truncation (sources just outside the window) is intrinsic and stays a known limit.

### H3 result (job 4005; 3257 harvested windows, third-order far field, 50% Exp E replay)
Verdict: KILL by the pre-registered rule (mid |beta| > 0.5); plain-scene criterion PASSES.
- Plain err@20 vs Exp E: .0387/.0406 vs .0371/.0395 (1.04x/1.03x). dE@100 .038/.033 vs .014 (2.4x). Replay fixed the H1 collapse (12.8x).
- Far-field expansion error (relative, mean over windows): order 2 about 0.30, order 3 0.24 overall (shards with few windows ~0.37, good shards 0.14). Truncation from sources just outside the window is largely intrinsic, third order helps modestly.
- Channel use: val loss with ext / channels zeroed / no ext = .0037/.0284/.0510, so the extapply field carries the gain; channels zeroed is 7.6x worse (the net now depends on the channels or on their consistency with extapply; untested which).
- Hole-4 beta, held-out, pooled ticks 5/10/20, bins 8-16 / 16-32 (Exp E: sparse -0.04/-0.22, mid +0.25/+0.38, dense +1.3/+1.8): sparse -0.27/-0.55, mid -0.94/-1.05, dense -0.33/-0.32. Mid is still strongly negative, so the mid-density bias is not fixed by harvested far-field-consistent windows either; A0 (no windows) had -0.50/-0.54.
- Multiclump to step 100: finite, max |v| 17, no blow-up.
- Open: why mid beta is negative in every fine-tuned variant while Exp E is positive (no variant tried keeps Exp E's training scenes unchanged and adds only ext; the common factor is fine-tuning at lr 2e-4 for 50 epochs). Candidate not tried: much shorter/lower-lr fine-tune, or freezing the net body and training only new inputs.

### Contact Bh2 (job 4006): bounded (tanh 5) + penalised (0.1) head
Verdict: KILL (diverged). Training loss left 4 at k=9 (it ~2000) and 5854 of 8000 steps were skipped by the loss cap, same pattern as the first B and A runs before bounded force heads. Eval: err@20 374-965, tunnel@100 100% on all six families, KE ratio 4e8-4e16, wall restitution 0.32-0.41. B2 baseline unaffected (err@20 4.1-5.1, tunnel 0). The soft penalty and tanh bound were not enough; the head output enters the force before the existing FMAX bound and may need to be bounded and zero for sub-threshold contact, or the head applied with lower lr / smaller init. Not tried.

### B1 tiling (job 4007)
`t0ext:armB1.pt` through `tiered_eval.py` gravity and seams groups: both `ok` (gravity 109 s). The earlier crash (hidden size 245 vs 244 between windows) is gone with the id-keyed store (Tiled passes the particle ids and an owner-tile write mask; commit 936af00 in worktree-agent-a8b09cd8f67d9fd44, needs scatter_tier0.set_ids from this branch). Results JSON on Polaris: ~/t0t/results/tiered/eval_t0ext_B1_ids.json; accuracy numbers not examined.

## Fine-tuning-itself hypothesis (pre-registered 2026-10-25, before launch)
Observation: mid-density beta (8-16 / 16-32, held-out) is negative in every fine-tuned variant (A0 -0.50/-0.54, A1 -0.80/-0.90, B1 -0.74/-0.84, H3 -0.94/-1.05) and positive for Exp E (+0.25/+0.38, 2 held-out seeds). Tests, one change each, all from Exp E, seed 4738:
- E5: Exp E untouched, beta on 5 held-out seeds (9000, 12000, 13000, 14000, 15000), sparse/mid/dense. Question: is the baseline's mid beta robust? PASS-as-baseline: mid beta within [+0.1, +0.5] at both bins pooled over 5 seeds and no seed negative; if seeds scatter around 0 or are negative, the "Exp E is positive" premise is noise.
- Fa: A0 scene mix (no ext), `halo=8`, lr 2e-5, 10 epochs (25x less update than A0). If mid |beta| <= 0.15 -> fine-tune size is the cause; if >= 0.5 -> any fine-tune on these scenes biases; between -> dose-dependent.
- Fb: body frozen, only new first-conv input channels trained (spec `ext=6,extapply,taper=8,bnd,halo=8`, ext scene mix of arm A, lr 2e-4, 20 epochs), start = Exp E exactly except taper. If mid |beta| <= 0.15 the bias comes from body updates; >= 0.5 -> from data/ext path even with a frozen body.
- Fc (LoRA adapter): not run; the first conv is the only part that takes new inputs, and Fb already restricts updates to it; a low-rank adapter on the body would need new net code. Listed as not done.
Scored: hole-4 beta all scenes, plain err@20 vs Exp E, dE@100. No pass/kill on the arms themselves (diagnostic); a variant is "clean" if |beta| <= 0.15 sparse+mid, <= 0.5 dense, err@20 <= 1.10x.

### Fine-tuning-itself results (job 4008; held-out seeds 9000, 12000, 13000, 14000, 15000, pooled ticks 5/10/20; bins 8-16 / 16-32)
- E5 (Exp E untouched): sparse -0.03/-0.19, mid +0.24/+0.39, dense +1.23/+1.62. Same as the 2-seed numbers (+.25/+.38), so the baseline is robust and not the outlier.
- Fa (A0 mix, halo=8, lr 2e-5, 10 epochs = 25x smaller update than A0): sparse -0.71/-0.54, mid -0.45/-0.42, dense +0.11/+0.26. Plain err@20 1.18x/1.17x Exp E (worse than the 50-epoch A0 at 1.00x), dE@100 .020/.018. Bias is NOT dose-dependent: a tiny fine-tune on this mix already flips mid and sparse negative. (Only fine-tune lr changed vs A0; isolated-mix training itself is the cause, in this setup.)
- Fb (body frozen, only the first-conv new input channels trained, ext scene mix, 20 epochs, lr 2e-4): sparse -0.24/-0.25, mid +0.40/+0.76, dense +0.99/+1.44. Beta structure stays Exp E-like (positive mid and dense); plain err@20 1.09x/1.07x, dE@100 .038/.034. Mid 16-32 +0.76 exceeds 0.5 and sparse -0.25 exceeds 0.15, so not "clean", but the sign flip is gone.
- Reading: the negative mid/sparse flip comes from updating the body weights on these scenes, even slightly, not from the ext path or data volume. Training only new input channels keeps Exp E's bias profile (including its positive dense/mid), and is therefore the safest way to add inputs; it does not remove the baseline's own bias (dense +1.0..+1.4).
- Not run: LoRA-style adapter (Fc). Fb is the stronger form of the same restriction for the input path; a body adapter needs new net code.

### B1 tiled accuracy (job 4007, results/tiered/eval_t0ext_B1_ids.json; harness groups gravity, seams)
- Seams (tiled vs exact rel rms): B1 with id-keyed hidden 0.1828, B1 earlier run (from D2's eval_t0ext_B1.json, which ran seams before the hidden state was keyed by id, per-window reset) 0.1820, single-window 0.2228, A1 for reference 0.0989 (single 0.1486). Seam jump / field 2.32 (earlier 2.28; truth 0.0037; A1 1.86). So keying the hidden state by particle id changes nothing measurable at seams: B1 tiling is as good as before, not better; both are about 1.8x worse than A1 on tiled-vs-exact error and far above truth on seam jump (ratio 260 vs truth 0.83).
- Gravity group (plummer): B1 tiled err@5/10/20/50/100 = .0044/.0150/.0517/.282/1.47; A1 (earlier, same harness) .0020/.0063/.0204/.091/.319. dE: B1 model drift .0026 at tick 1 rising to .027 by tick 20; A1 .00005..., so B1 gravity tiling runs but drifts about 50-500x more in energy than A1 and is 2.5-4.6x worse in position error. No untiled B1 gravity number exists (that group crashed before), so tiling cost itself cannot be isolated; the gap is the B1 model (drag-trained head) on gravity, not tiling.

## Round 2026-10-25b results (job 4020)
(0) Positive-feedback hypothesis REFUTED for Bh4: mean head power on active particles is negative throughout (-0.11 at step 1 to -0.66 at step 31), fraction with positive power 0-8% until step 19, rising to 15-55% only at steps 22-31 (just before the blow-up), then the system is already at speed 70-4000. The head does not pump energy steadily; the blow-up starts when speed passes the trained range (>15) at wall contact, with the head turning from dissipative to mixed sign. "Negative damping" is withdrawn as the mechanism; the trigger is out-of-range contact states.
(1) Bh5 (gated head, k 4 -> 60): KILL. Training unstable (6148 steps skipped by the loss cap), err@20 69-84, tunnel@100 14-41%, KE ratio@100 5.9-64 (outside [0.1, 10] on 5/6), wall restitution 0.71/0.81 (B2 0.49/0.57, truth 1.0). Long-horizon training makes it bounded (no 1e5+ blow-up) but not accurate. Bh6 (head only on frozen B2, k 4 -> 60, 4000 its, lr 1e-3): stable and exactly B2-like: err@20 4.13-5.1 (<= 1.00x B2), tunnel@100 0 on all, KE ratio .0025-.0075 (B2 .0027-.0086), restitution .49/.57 unchanged. KILL by the KE rule = "safe, no gain": the head learned nothing useful (residual of B2 is small relative to the head's penalties). Pick: Bh6 is the only usable arm and equals B2; no contact benefit from a per-particle head so far.
(2) Fine-tune bias, 5 held-out seeds (8-16 / 16-32; Exp E: sparse -0.03/-0.19, mid +0.24/+0.39, dense +1.23/+1.62; Fb: mid +0.40/+0.76):
- FbH3 (Fb + harvested windows + 50% replay + 3rd-order far field): sparse +0.00/+0.01 (clean), mid +0.60/+1.02, dense +1.01/+1.44. Plain err@20 1.65x/1.59x Exp E, dE@100 .015. Not clean: mid > 0.5, err@20 well over 1.10x.
- FbL (rank-4 LoRA on every conv + new inputs, Fb scene mix): sparse -0.65/-0.73, mid -1.29/-1.35, dense +0.96/+1.35. Plain err@20 1.49x/1.41x, dE@100 .020/.015. Worse than Fb: the adapter on the body reproduces the sign flip (updating the body, even low-rank, flips mid), supporting that body updates cause it.
- So far no variant is clean on all of sparse/mid: only the frozen-body variants keep mid positive; FbH3 fixes sparse (0.00) at the price of err@20 and a larger positive mid bias.
(3) B1 gravity (Plummer N=300, seed 4738, 2 scenes), untiled span 64 / tiled span 128: err@20 | err@100 | dE@100 (rel): B1 untiled .059|1.33|.31, tiled .081|1.83|.013; B1 head zeroed untiled .027|.43|.040, tiled .075|1.81|.33; A1 untiled .026|.47|.029, tiled .050|1.28|.15; Exp E untiled .037|.41|.047, tiled .185|4.32|2.09 (truth dE ~1e-6). Head ablation: zeroing the head restores B1 to A1-level untiled accuracy (err@20 .027 vs .059, dE@100 .040 vs .31): the drag-trained head is the source of the extra error and energy drift on gravity (it applies drag to non-drag particles; the material channel is zero here). Tiling cost for B1: err@20 x1.4 (.059 -> .081), err@100 x1.4 (still E: x4.6 tiled), consistent with A1 x1.9 and far below Exp E x4.9: the tier-0 variants tile better than plain Exp E at 128 span.
