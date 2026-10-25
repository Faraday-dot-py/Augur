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


### Hole 1 test outcomes (2026-10-24, branch hole1-potential-field in worktree agent-adfbf776853ca7eea; full numbers in docs/debugging/experiment-log.md)
- LJ tier-0 test: mesh force = -autograd gradient of a scalar energy of density only (+ pair term + global mean removal): 160-frame NVE |dE/N| <= .0084 on seeds 4738/4739/4740 vs -.26/-.29 (ft liquid/solid): PASS. Adding the soft-penalised residual net (magnitude + curl, 300 each): |dE/N| <= .0093, penalty keeps residual ~1e-4 (off); no gain, no leak. Full eval: no scene T > 3% off; remaining error is the pair term (matches pair-only reference).
- Caveat: the energy mesh is also memoryless and position-only; gradient structure and removal of velocity/history inputs are not separated. A velocity-dependent term would have to live in the residual channel, and the test did not show whether the penalty can contain a velocity-using residual that is actually needed (nothing in LJ needs one).
- Polynomial-cell potential (phi0 + g.d + 0.5 d^T H d, bilinear blend, force = autograd gradient): exactly conservative (loop work ~1e-5 vs 1e-3..7e-3 relative for blending per-cell forces); order 2 beats order 1; held-out force error 1.7-2.5x the naive blend at sigma >= 1.5 in one fit, 0.9-1.6x in another (noisy). Accept the gradient form; accuracy cost unresolved.

## Measured baselines (2026-10-24, existing Exp E + R1b + TiledScatter, no training; details experiment-log.md "Hetero-tiered baselines RESULTS")
### Hole 4 (window bias) -- inconclusive, leaning no mean-field absorption at in-distribution density
- Exp E was trained on isolated scenes of extent 64, i.e. already the hole-3 regime minus the external-field input. Run on 64x64 windows cut from larger scenes (teacher-forced exact states, carried field), beta = <(model - isolated truth).(full - isolated truth)>/<F.F>.
- Sparse (~70 bodies/window): beta 8-16 from edge -0.03 (.04), 16-32 -0.20 (.10), 4-8 +0.00; bias as fraction of |a| -0.4% / -1.5%. Mid (~270): +0.25 (.05) / +0.38 (.19), one seed ~0. Dense (~700, OOD): +1.3 / +1.8, both seeds (KILL pattern but density-confounded).
- Limit: model's own isolated-force error (15% sparse) exceeds the far-field size (7-12% of |a|); detection power is low. Edge 0-4 bin error 175% (edge artifact), cold-field tick 0 error ~50%.
- Not tested: a tier 0 trained with the hole-3 external-field channel and truth-minus-external target.
### Hole 5 (teacher-forced vs rollout) -- PASS at ticks <= 20
- Tier-1 target proxy R = a_exact - a_tier0 (halo-only tier 0 = g16_fx): relative |R| .10-.13, 85-90% tile-scale smooth, inward (missing far pull). Rollout/teacher-forced ratio 1.00/1.01/1.04/1.12 at ticks 5/10/20/40 (smooth part 1.00/1.01/1.07/1.22); 4x4x32 1.05/1.13 at 20/40. With far term 0.99-1.00. cold300 0.95-1.01. Teacher forcing suffices for horizons <= 20; a small gap opens by tick 40.
### Hole 7 (seams) -- PASS
- Seam-crossing energy-change excess and accel-error-jump excess: tiled equals single within SE (e.g. de excess -.087(.023) vs -.090(.019); jump +.033 vs +.027); drift cluster identical to 3 digits; position error identical to 3 digits in the frame dump. Single-grid virtual seams show the same excess (speed effect, not seam). dE@40/KE0 tiled -.0125 vs single +.0139. This validates the ghost-halo + coarse far path only; the potential-blend seam mechanism of hole 7 was not exercised (no tier >= 1 exists).

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

## Implementation status: tiered data/eval harness (2026-10-24, branch worktree-agent-a8b09cd8f67d9fd44)
Code: scripts/tiered_{gen,scenes,iface,eval,adapters,arms,summarize}.py, tests/test_tiered_harness.py (renamed from test_tiered.py to avoid clash with C's tests/test_tiered.py). 12-group eval harness, cell-multipole truth rel rms 5e-5 vs direct sum, branch adapters for the hole-9 arm driver. Baselines on Exp E ckpt (4 seeds): plummer err@20 .030, multiclump err@100 6.43, bias z_x -2.95, seam jump/field 4.56 vs truth .004; far-dom force err .438/.443 (A/B), N=2000 2.5/8.4 ms. Details: docs/debugging/experiment-log.md "Tiered-architecture data generators" and "Tiered harness round 2".

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

## Tier-0 head on LJ: arm PH (pre-registered 2026-10-24, before launch)
Arm P (hole1-potential-field): energy-gradient density-only mesh + learned pair term, scratch, 6000 its, seed 4738: |dE/N| <= .0093 over 160 frames, three seeds. Arm PH = P plus the tier-0 per-particle head (`scatter_tier0`, head=32, no hidden state: tile-packed tokens have no persistent identity here, untested) added to the total acceleration before global mean removal. Head inputs: own velocity, velocity relative to the local mean flow, |rel|, mass, log density, total (mesh+pair) acceleration; zero-init output (identity at start); soft penalty on head acceleration magnitude with the existing lam-mag 300 (the same soft-penalty rule as arm PR's residual; no hard-coded conservation). One change vs P, identical protocol (`scripts/polaris_scatter_lj_h1.sh ARM=PH`, `--energy --head 32`), then the same drift + six-scene eval.
PASS: all six scenes meet P's gate (T within 5%, psi6 within .03, P within 15%), |dE/N| <= .02 over 160 frames in the four NVE scenes on seeds 4738/4739/4740 (2x P's worst), and err@20 <= P's in >= 4/6 scenes (head not harmful; helping is a bonus).
KILL: any scene T off > 5%, or |dE/N| > .05 on any seed, or err@20 > 1.5x P in >= 4/6 scenes. Otherwise inconclusive. Contact balls: separate arm after this one.

### Arm PH result (job 3995, 6000 its, 45 min train, seed 4738; P in parens)
Verdict: INCONCLUSIVE (gate passes; err@20 criterion fails; no kill trigger).
- Gate: all six scenes finite, T within 5% (dT liquid .013, coexist .026, quench .002, solid .021, melt .001, gas .007), psi6 within .01, P within 15% (quench P error .010 vs .231 for P; others <=.026). Pass.
- Drift |dE/N| over 160 frames, 3 seeds (P): gas .002-.004 (.002), liquid .0002-.0012 (.0006-.0011), solid .0005-.0007 (.0016-.0018), coexist .0103-.0116 (.0073-.0084). Max .0116 <= .02. Pass. Head did not reintroduce drift.
- err@20 (P): liquid .174 (.148), coexist .165 (.118), quench .352 (.341), solid .0178 (.0171), melt .0102 (.0093), gas .0076 (.0034). Worse than P in 6/6 (1.0x-2.2x; kill needs >1.5x in >=4/6: only gas 2.2x, coexist 1.4x). Criterion "err@20 <= P in >=4/6" fails.
- Reading: with the soft penalty (lam-mag 300) the head stays harmless on conservative LJ but adds nothing (nothing velocity-dependent in the target), slightly hurts short-horizon accuracy; the quench pressure gain is a side effect to check against seed noise. Hidden state not used. Raw: results/scatter_lj_h1_PH.json, results/scatter_lj_drift_PH.json.

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

## Harvest arms H1 / H2 (pre-registered 2026-10-24, before launch)
Data: windows (64x64) cut from large softened-gravity rollouts (`tiered_gen.py harvest`: 8192 bodies, domain 256, IC mix clumpy/plummer/disc/cold_collapse/uniform, 24-step windows, n-max 300, 63 shards x 32 windows = ~2000 windows, seed 4738); per-frame direct-sum far-field expansion (a0, J) of everything outside the window; truth positions include the far force. Quadratic term and bodies entering the window after frame 0 are dropped (expansion error logged). This replaces the isolated-scene mix that gave mid beta -0.50 (A0).
Both arms: Exp E warm start, taper=8,bnd,halo=8, lr 2e-4, 50 epochs, k 4->20, NaN rollback, seed 4738.
- H1: `ext=6,extapply`: far field applied exactly, net learns the near field (ext channels also fed, unscaled).
- H2: `ext=6` with no extapply and ext channels scaled to unit rms (`--norm-ext`): net must use the channels to reproduce the far force. Usage test: val loss with channels zeroed >= 2x val loss with them.
Eval (ext fed zero = tier 0 alone): hole-4 beta on sparse/mid/dense (held-out seeds, ticks 5/10/20), plain-scene err@20 vs Exp E, multiclump rollout to step 100 (`tiered_diag_blowup.py`, via D's `t0ext:` loader).
PASS (per arm): |beta| <= 0.15 at 8-16 and 16-32 on sparse and mid, <= 0.5 dense, plain err@20 <= 1.10x Exp E, multiclump finite with max|v| < 50 to step 100. KILL: |beta| > 0.5 sparse or mid, plain err@20 > 1.25x, or multiclump blow-up. Else inconclusive.

## Contact balls with the tier-0 head: arm Bh (pre-registered 2026-10-24, before launch)
Reuses the scatter-contact training/eval code (`scatter_contact_scenes.py`, ContactScatterField, SDF obstacle channels, bounded force heads FMAX 40) unchanged except `--t0 head=32` (tier-0 per-particle head, zero-init, no hidden state). Recipe identical to B2 (job 3380): Exp E init, 8000 its, lr 1e-4, k 4->20, 800 training scenes x 80 steps, seed 4738, eval seeds 9100/9200/9300 on box15, box_var, post_circle, post_rect, held-out disc, post_poly. B2 is the baseline (err@20 4.1-5.1, tunnel@100 0.0% everywhere, KE ratio@100 .003-.009 = over-damped, restitution .85-1.15). Old option C (c_id err@20 5.10) is a different family set, compared in the text only.
PASS: tunnel@100 <= 1% on all six families, err@20 <= 1.10x B2 on all six, and mean KE ratio@100 >= 3x B2's (the head sees velocity and can supply the missing dissipation/energy balance). KILL: any tunnel@100 > 5%, err@20 > 1.25x B2 on any family, or divergence. Otherwise inconclusive.

### Harvest arms H1 / H2 result (job 3997, 1718 harvested windows, 50 epochs, lr 2e-4, no NaN)
Verdict: both KILL (plain err@20 >> 1.25x Exp E; sparse beta far over 0.5).
- Data: far-field expansion (a0, J) leaves a mean relative residual of 0.30 (0.18-0.68 per shard) of the far force; rms a0 0.067, rms J 0.0019.
- Plain isolated-scene err@20 (Exp E .0371/.0395): H1 .476/.546 (12.8x/13.8x), H2 .878/.952 (23.7x/24.1x); err@100 12 vs 2.8. Training on windows only destroyed isolated-scene accuracy (no Exp E replay in the mix).
- Hole-4 sparse beta pooled 8-16 / 16-32 (ext fed zero): H1 +1.33/+0.64, H2 +2.05/+1.26 (Exp E -0.04/-0.22). Mid and dense summaries were not captured by the log filter; raw json in ~/t0h/results/tier0_ext/hetero_a_H{1,2}_*.json on Polaris.
- Channel use (val windows, loss with ext / channels zeroed / no ext at all): H1 .0048/.0052/.0184 (channels ignored, extapply carries it); H2 .0135/.0175/.0206 (zeroing hurts only 1.3x; without extapply the net recovers only ~1/3 of what exact application gives, so the channels are weakly used, below the 2x usage criterion).
- Multiclump to step 100 (D2 blow-up diagnostic): both finite, max |v| <= 26 (H1) and <= 15 (H2), no blow-up; first position error >10 at steps 24-56. Armed-A style divergence is absent at lr 2e-4.
- Reading: harvested-window training with a truncated (30% residual) far field overfits the window distribution and costs isolated accuracy; not a viable replacement. Untested: mixing harvested windows with Exp E isolated scenes (replay), higher-order far expansion, larger window set (1718 windows is small).

### Contact arm Bh result (job 4001, 8000 its, 23 min, B2 recipe + head=32)
Verdict: KILL (tunnel@100 98-99% on all six families; pre-registered kill is any > 5%).
- err@5/10/20 match B2 (box15 4.98 vs 5.11, box_var 4.47 vs 4.53, post_circle 4.92 vs 4.92, post_rect 4.14 vs 4.11, disc 4.36 vs 4.49, post_poly 4.82 vs 4.83): within 1.01x everywhere. Truth tunnel@100 <= 1.5%.
- Beyond the training horizon (k_end 20) it explodes: err@50 45-66, err@100 5e3-2e7, KE ratio@100 1e7-7e14, tunnel@50 72-82%. B2 stays at err@100 3.8-5.5, KE .003-.009, tunnel 0. Same failure signature as arm A2 (kernel option), now with the head: a velocity-dependent per-particle force is unbounded past the trained horizon, and the head has no energy-closing term (penalty or FMAX bound; B2's bounded heads do not cover it).
- Wall restitution 0.46 (v=12), 0.66 (v=15) vs B2 0.49/0.57, truth 1.0. No KE-ratio gain (B2's over-damping unchanged until blow-up).
- Not tried: tanh-bound / soft magnitude penalty on the head (as in LJ PH, where lam-mag 300 kept it harmless), training with longer k_end, head fed only relative velocities.

### Regression after set_ids (job 4003, Polaris CPU)
`scatter_tier0_tests.py --device cpu --ckpt E --skip-train`: ALL PASSED, including flags-off bitwise vs pre-change on the real E checkpoint and all synthetic configs; tests/test_tier0_ids.py passes. The earlier local failure was the local machine's numerics against a Polaris-made reference, not the code.

## Arm H3: harvested windows + Exp E replay + third-order far field (pre-registered 2026-10-24, before launch)
Change from H1 (killed, plain err@20 12.8x): (a) each step's loss is 0.5 x windows + 0.5 x isolated Exp E scenes (1500, 40 ticks, `--replay 0.5`), so isolated accuracy is kept; (b) 120 shards x 32 windows (target ~3800) with the far field expanded to THIRD order (a0, J, Q, C; `scripts/tier0_harvest.py`, expansion about the window centre, sources outside the 64x64 window) and applied exactly by extapply (set_external `high=(Q, C)`); the second-order expansion left 0.30 relative error (H1), the harvest logs error at order 2 and 3. Everything else as H1 (ext=6,extapply,taper=8,bnd,halo=8; Exp E warm start; lr 2e-4; 50 epochs; k 4->20; seed 4738). Hole-4 table now from `scripts/hetero_beta_table.py` (held-out seeds, all three scenes).
PASS: |beta| <= 0.15 at 8-16 and 16-32 on sparse and mid, <= 0.5 dense, plain err@20 <= 1.10x Exp E, multiclump finite with max|v| < 50 to step 100. KILL: plain err@20 > 1.25x Exp E, or |beta| > 0.5 sparse or mid, or multiclump blow-up. Else inconclusive. If the third-order error is still > 0.15 the far-field truncation (sources just outside the window) is intrinsic and stays a known limit.

## Contact arm Bh2: bounded + penalised head (pre-registered 2026-10-24, before launch)
Change from Bh (killed: tunnel@100 98-99%, KE 1e7+ after tick 30): the head output is bounded, a = 5 tanh(a_raw/5) (`headmax=5`), and a soft penalty 0.1 x mean-square head acceleration is added to the loss (`--head-pen 0.1`; scaled from LJ PH's 300 at accelerations ~0.1 to contact accelerations ~5). Soft penalty and bound only, no hard-coded conservation. Everything else is B2/Bh (Exp E init, 8000 its, lr 1e-4, k 4->20, 800 scenes, seed 4738); spec `head=32,headmax=5`.
PASS: tunnel@100 <= 1% and KE ratio@100 in [0.1, 10] on all six families, err@20 <= 1.10x B2. KILL: tunnel@100 > 5% or KE ratio@100 outside [0.1, 10] on any family, or err@20 > 1.25x B2. Else inconclusive. B2: err@20 4.1-5.1, tunnel 0, KE .0027-.0086, wall restitution .49 (v=12)/.57 (v=15).

## Contact head divergence diagnosis (jobs 4011/4013, 2026-10-25) and arm Bh3 (pre-registered before launch)
Diagnosis (scripts/contact_head_diag.py, post_circle eval scenes, 30 rollout steps):
- Step-0 identity: zero-init head model equals the same model without it exactly (max abs diff 0.0), so the divergence is learned, not an initialisation bug.
- Bh (unbounded head, job 4001): head acceleration is large away from contact. Mean head |a| 0.7-2.6 over the rollout, max 4-16, with the max-output particle at 1.2-2.7 surface-gap units from any surface and speed < 1 in several steps; only the last logged step is at a wall (sdf 0.13, speed 15.6, head 16). Training speed percentiles: p50 6.1, p99 15.4, max 22.5.
- Bh2 (tanh-5 head, penalty 0.1, diverged run's final weights): head saturates at 5*sqrt(2)=7.07 (both components on the bound) for most particles; speeds reach 35 at step 4 and 748 at step 7, particles end 20 to 2300 units inside walls (sdf negative). The bound turns the head into a constant-magnitude push, and the penalty (0.1 x mean|a|^2 ~ 0.1 x 25) is small against losses of 1e3+.
- Cause: the head is not confined to contact. In B the learned wall and pair terms are windowed by gap (`_win`, range 2); the head is global, sees only mesh acceleration and velocity (no contact force, no geometry), and is used to absorb the mesh branch's error in free flight, which is out of its trained velocity range once the rollout leaves the training horizon.
Fix (architectural; no hard conservative force, no speed cap, no impulse): (i) gate the head output by the same contact window as the contact terms, gate = _win(min(wall gap, nearest-neighbour gap), 2), so free flight is exactly B2 (`headgate`, `ContactScatterField.contact_gate`); (ii) clamp the head's velocity inputs to the training range [-15, 15] (`vclip=15`); (iii) keep the tanh-5 bound and the 0.1 soft penalty.
Arm Bh3: B2 recipe (Exp E init, 8000 its, lr 1e-4, k 4->20, seed 4738), spec `head=32,headmax=5,headgate,vclip=15`, `--head-pen 0.1`. PASS: tunnel@100 <= 1% and KE ratio@100 in [0.1, 10] on all six families, err@20 <= 1.10x B2. KILL: tunnel@100 > 5% or KE ratio outside [0.1, 10] or err@20 > 1.25x B2 on any family. B2 KE ratio is .003-.009, so a pass needs the head to supply the dissipation B2 over-does; if Bh3 merely equals B2 (KE .003-.009, inside no band) it is KILL by the KE rule although safe, and will be reported as "safe, no gain".

### Bh3 result (job 4014) and second diagnosis (job 4017)
Bh3 (gated + clamped + tanh-5 + 0.1 penalty): training stable (0 skipped steps), err@5/10/20 equal to B2 or better (box15 1.61/2.68/4.96 vs B2 1.70/3.16/5.11; all six families <= 1.01x B2 at tick 20), but beyond the training horizon it still blows up: tunnel@100 100%, KE ratio@100 1e6-3e9, err@100 1.3e4-7.5e4. KILL (pre-registered: tunnel > 5%).
Diagnosis (60-step rollout, post_circle eval scenes): head output small while the gate is open only near contacts (mean 0.03-0.2, max < 0.5 through step 34). At step 37-43 it rises (max 0.7 -> 1.8) on a particle at the wall (sdf ~0 to 0.4) moving at 10-33; by step 46 that particle is at speed 3110 and 18 units inside the wall, and the head saturates (7.07). So the failure starts at wall contact after the training horizon: the velocity-dependent head learned positive feedback (force aligned with velocity = negative damping, supplying the energy B2 over-removes) and gains energy exponentially; the wall term (bounded by FMAX 40) cannot hold it.
Next arm Bh4 (pre-registered): Bh3 plus a one-sided soft penalty on head power, 1.0 x mean(relu(a_head . v)) (`--head-pow 1.0`): the head may dissipate or turn the velocity but is penalised for adding kinetic energy per particle. Soft penalty only, no hard conservative or speed-cap element. Same recipe otherwise. PASS: tunnel@100 <= 1%, KE ratio@100 in [0.1, 10] on all six families, err@20 <= 1.10x B2. KILL: tunnel > 5% or KE outside [0.1, 10] or err@20 > 1.25x B2. Prediction: tunnel passes; KE stays near B2's .003-.009 (the penalty forbids the energy supply that the KE band needs), so the likely outcome is "safe, no gain" = KILL on the KE criterion.

### Bh4 result (job 4018): Bh3 + one-sided power penalty (1.0)
KILL. Training stable (0 skipped), err@20 within 1.01x of B2 on all six families (4.2-5.05), but tunnel@100 100% and KE ratio@100 5e4-5e8 (err@100 9e3-7e4) on all. The power penalty did not prevent blow-up past the training horizon: the penalty is only active on states seen in k<=20 rollouts, where the head is nearly inactive, so nothing teaches it behaviour at post-horizon wall contact. Contact head arms Bh, Bh2, Bh3, Bh4 all: err@20 equal to B2, no benefit, long-horizon explosion (B2 has none). Untried: training with k_end > 40 (wall-contact states after the first bounce), head trained on frozen B2 residuals only, dropping the velocity input (position/gap only). Conclusion for now: a velocity-dependent per-particle head is not usable in the contact model without longer-horizon training; it is usable on LJ (PH, penalised, gate-free) and on drag-gravity (B1).

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

## Round 2026-10-25b (pre-registered before launch; one chained Polaris job, one GPU job at a time)
(0) Positive-feedback test: `contact_head_diag.py` now logs, per rollout step, mean head power a_head.v over active particles (|a_head| > 1e-3) and the fraction with positive power, on the Bh4 checkpoint (60 steps). Hypothesis (from Bh3 diagnosis): positive-power fraction > 0.7 and growing mean power at the wall before the blow-up. If not, the "negative damping" explanation is wrong and is withdrawn.
(1) Contact arms, order Bh5 then Bh6, each vs B2 (err@20 4.1-5.1, tunnel 0, KE .003-.009):
- Bh5: Bh4 recipe (spec `head=32,headmax=5,headgate,vclip=15`, head-pen 0.1, head-pow 1.0, Exp E init, lr 1e-4) with training horizon k 4 -> 60 (`--k-end 60`, linear curriculum), 8000 its. Tests whether post-horizon wall states in training stop the explosion.
- Bh6: head only, B2 weights frozen (`--init B2.pt --head-only`), same head spec and penalties, k 4 -> 60, 4000 its: the head learns the residual of the frozen B2 model, so it cannot degrade B2's mesh/contact terms.
PASS (either): tunnel@100 <= 1% and KE ratio@100 in [0.1, 10] on all six families, err@20 <= 1.10x B2. KILL: tunnel > 5%, or KE ratio outside [0.1, 10], or err@20 > 1.25x B2. "Safe, no gain" (tunnel 0, KE in B2's .003-.009) counts as KILL by the KE rule and is reported as such. Pick by evidence; no third arm without a new diagnosis.
(2) Fine-tune bias, from Fb (body frozen, new first-conv channels only; 5-seed beta mid +0.40/+0.76, sparse -0.25, dense +0.99/+1.44, err@20 1.09x/1.07x, dE@100 .038/.034):
- FbH3: Fb with harvested windows + 50% Exp E replay + third-order far field (H3 data/recipe, `--freeze-body`, lr 2e-4, 20 epochs).
- FbL: Fb scene mix with a rank-4 LoRA-style adapter on every grid-net conv (`lora=4`, zero-init output, base weights frozen, adapter + new input channels trained; lr 2e-4, 20 epochs).
Scored on 5 held-out seeds (9000, 12000, 13000, 14000, 15000): hole-4 beta sparse/mid/dense, plain err@20 vs Exp E, dE@100. Clean = |beta| <= 0.15 on sparse and mid (both bins), <= 0.5 dense, err@20 <= 1.10x. No kill rule (diagnostic); a variant that is worse than Fb on mid and sparse |beta| is dropped.
(3) B1 gravity: untiled (span 64, one window) and tiled (span 128) Plummer N=300 rollouts, seed 4738, 100 steps, for B1, B1 with the head zeroed, A1, Exp E (`tier0_b1_ablate.py`): err@20/100 and relative energy drift at ticks 20/100. Separates tiling cost (tiled - untiled) from model cost, and the head's contribution to the 50x dE (B1 vs B1-nohead).
