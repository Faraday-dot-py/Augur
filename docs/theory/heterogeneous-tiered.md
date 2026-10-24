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
