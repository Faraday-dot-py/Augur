# CFD contact/collision generalization (2026-09-28)

Design only; nothing implemented, no compute run.

## 0. Goal

`CentralForceDynamics` (`model/central_force.py`, checkpoints `gravity_central_v1.pt` / `gravity_relativistic_central.pt`) is CFD's general-purpose, O(N), tileable, interpretable force model, but its force is a scalar function of center-to-center distance only, `f(d)`, applied along the line between centers. It has no notion of body size, so it cannot represent contact: a ball bouncing off another ball, a ball bouncing off a wall, or — the actual target — a ball bouncing off an obstacle of arbitrary shape dropped into the scene *after* training, with no obstacle-specific code path.

This spec extends CFD so the same interaction mechanism covers gravity, ball-ball contact, and ball-obstacle contact for obstacles of arbitrary rigid shape, while keeping CFD's core properties (O(N)/tileable, exact momentum conservation, interpretable pairwise force).

**Phasing (resolved 2026-09-28):** built in two phases. **Phase 1** (this implementation) is normal-contact-only: mass/radius/kinematic state, the conservative `V(d, r_i+r_j, s_ij)` force, and obstacles as circles and flat walls placed anywhere in the scene. **Phase 2**, deferred but kept as an explicit future goal, is spin and the tangential/friction force (Section 3). Nothing in Phase 1's architecture blocks adding Phase 2 later — spin is an additional per-body channel and the tangential force is an additional term applied at the same edges, not a redesign.

## 1. Prior resolved decision this design reuses

A 2026-09-26 architecture discussion (session `762042ce`, "Block A / Block B" design for the token model) already resolved the two questions that matter here, and this design carries both forward rather than re-deriving them:

- **Geometry-as-input is only needed if geometry varies at test time.** A fixed, trained-in environment can live entirely in learned weights; feeding geometry in as a runtime input is a separate, heavier requirement. Here geometry *does* vary at test time (arbitrary obstacle shapes dropped in post-training), so this design is in the "feed geometry in" branch deliberately, not by default.
- **Symmetric attention is possible and is the right way to make a pairwise force depend on per-body attributes.** Per-destination softmax breaks Newton's third law (asymmetric edge weight `w_ij != w_ji`); the fix is a symmetric score `s_ij = s_ji` (no per-destination normalization), force along the pair line `F_ij = w_ij * phi(d) * unit_vector`, and — for energy conservation — the force should be the gradient of a symmetric pair potential `V_ij = w_ij * v(d)`, so `F_ij = -∇_d V_ij` and `F_ij = -F_ji` exactly. Per-ball attribute dependence (mass, and now radius/spin) was confirmed as a goal, which is what justifies using a learned `s_ij` instead of a bare `f(d)`.

This design's force law is exactly `F_ij = -∇_d V(d, s_ij)` instantiated with the attribute set this task needs (mass, radius, spin, kinematic flag), plus a separate non-conservative tangential term (Section 3) that the potential formulation deliberately does not cover.

## 2. Per-body state

State grows from `(pos, vel)` to `(pos, vel, mass, radius, spin, kinematic)`.

- **mass, radius, spin**: instantaneous per-body scalars (spin is a scalar angular velocity in 2D; a 3D version would need a vector, out of scope here). Within one training scene all real (non-obstacle) bodies share the same values, but those values vary across the training distribution (different scenes use different mass/radius/spin) — consistent with how mass was already scoped in `docs/debugging/z-and-mass-channels-feasibility.md` §mass channel, option A/B. No per-body-within-a-scene heterogeneity is required for real bodies; obstacles are the one source of heterogeneity in a scene (Section 4).
- **kinematic** (bool): if set, the body's position is driven externally (or held fixed) and is *not* updated by the integrator from received force — but it still exerts force on other bodies normally. This is the mechanism for "I can move a wall, but particle impacts can't move it."
- Inertia is **structural**: `a_i = (sum_j F_ij) / m_i` for non-kinematic bodies, per the z-and-mass-channels doc's recommendation (Newton's second law as a hard prior, not learned).

## 3. Force law

Per edge `(i, j)` from the existing radius graph (`model/token_graph.py`, unchanged — still O(N) via cell list):

- **Symmetric attribute coupling**: `s_ij = g(a_i, a_j)` where `a = (mass, radius, spin)` and `g` is swap-symmetric (e.g. computed on `(a_i+a_j, a_i*a_j, |a_i-a_j|)`, or an attention score with shared projections and no per-destination softmax, per Section 1). This is the direct reuse of the resolved "symmetric attention" design.
- **Normal (conservative) force**: `F_n = -∇_d V(d, r_i+r_j, s_ij)`, applied along the center-to-center unit vector, equal and opposite on `i` and `j`. `V` is a learned potential, one network, shared across the gravity/contact regimes: far from contact (`d >> r_i+r_j`) it should learn the existing smooth attractive gravity tail (this is exactly `CentralForceDynamics.force` reparameterized to take `r_i+r_j` as well as `d`, so `r_i=r_j=0` for all bodies recovers the current point-mass gravity model as a strict special case); near `d ≈ r_i+r_j` it should learn steep repulsion (contact). Momentum and energy are conserved by construction, same proof as `cons_pure`'s pair force and the existing `CentralForceDynamics`.
- **Tangential (non-conservative) force — Phase 2, deferred, not built now:** `F_t = h(d, r_i+r_j, v_rel_tangential, spin_i, spin_j)`, perpendicular to the center line, active only near contact (the network is free to learn to zero it out elsewhere). This is friction/rolling coupling and is *not* required to conserve energy — real friction dissipates kinetic energy, so a purely conservative model would be physically wrong here. It **is** required to satisfy Newton's third law (`F_t` on `i` is `-F_t` on `j`, by construction — same "apply to both edge endpoints" mechanism as the normal force) so total momentum is still exact even though energy is not. `F_t` also produces a torque on each body's spin (`τ = r_i * |F_t|`, sign from the tangential direction), which is the spin-update term in the integrator. Phase 1 has no spin channel and no tangential term at all — this bullet describes the intended future shape, kept here so Phase 2 has a design to build against rather than starting cold.
- **Why decompose into normal + tangential rather than one free-direction learned force vector**: a message with an unconstrained direction breaks angular momentum conservation even with symmetric weights (this was flagged explicitly in the 2026-09-26 discussion). Keeping the normal part central and isolating the tangential part as the only non-central term keeps the failure mode localized and named, instead of silently losing angular momentum everywhere.

Integrator: velocity Verlet, as today, extended to carry spin state and to divide accumulated force by `m_i` before the position/velocity update (kinematic bodies skip this update entirely).

## 4. Obstacles as boundary point clouds

An obstacle of arbitrary rigid shape (circle, wall, polygon, mesh boundary — anything) is discretized into a set of `kinematic=True` nodes sampled along its boundary, each carrying its own `radius` (sets the effective surface offset used by the same `V(d, r_i+r_j, s_ij)` normal-force term) and `mass = inf` in effect (never receives an update; `kinematic` flag makes this exact rather than approximate). These nodes are injected into the same radius graph real bodies already interact through — no new geometry subsystem, no SDF machinery, no per-obstacle-type code path. A body colliding with an obstacle is, from the force law's point of view, identical to colliding with another body that happens not to move.

This was chosen over an analytic-SDF obstacle representation (feeding `(SDF value, ∇SDF direction)` per particle) specifically because it reuses 100% of the existing radius-graph/cell-list/tiling infrastructure and keeps one interaction mechanism for gravity, ball-ball contact, and ball-obstacle contact, rather than adding a second, geometry-specific code path alongside it. The architecture itself (this section) stays general — any shape, discretized finely enough, works at inference — but Phase 1 *training data* only covers two primitives (Section 6): the point-cloud representation is what has to generalize to shapes never seen in training, not the training data itself.

Obstacle motion ("I can move a wall") is scripted position input, orthogonal to this design — same as any other kinematic trajectory. What can't move an obstacle is particle-exerted force; that's exactly `kinematic=True`.

### The boundary-sampling question, explained

This is a question about how an obstacle is *presented to the model* (as input, both at training time and at inference), not about how the truth physics is computed — truth stays exact/analytic (Section 6). Because the model only ever sees the world through per-node edges in a radius graph, a circle or a wall has to become a finite set of point samples along its boundary before the model can look at it at all. That turns one geometric object into three coupled tuning questions:

1. **Density (spacing between adjacent boundary points).** Too sparse, and a fast-moving ball can pass through the gap between two points in a single step before either one's contact radius engages it — the same failure mode already on record in this codebase for ball-ball contact when the neighbor radius is smaller than a fast body's per-step displacement (`docs/debugging/z-and-mass-channels-feasibility.md`). Too dense, and edges/compute are spent with no accuracy gain once spacing is well under the contact radius — pure waste, and it works against the O(N) argument if obstacles get arbitrarily fine for no reason.
2. **Per-point radius.** Adjacent boundary points' contact radii need to overlap enough to present a continuous surface (no ball-sized gaps between two points that are each individually solid) — so point radius and point spacing aren't independent choices, they're one joint "how continuous does this surface look" setting.
3. **Fixed vs. swept density in training.** If training data always uses one fixed sampling density, the model may implicitly learn to rely on that exact spacing and fail on a differently-sampled obstacle at inference (e.g. a downstream tool, or a user-drawn shape, that samples coarser or finer than training did). Sweeping density as a nuisance variable during training is the fix, at the cost of a bigger data-generation sweep — this is the same bias-to-a-fixed-setting risk as any nuisance variable left unswept, not something specific to obstacles.

None of this has a correct answer in the abstract — it's an empirical tradeoff (fidelity/robustness vs. compute) that this spec deliberately leaves as a tuning probe for implementation (Section 8), not a number decided up front.

## 5. Backward compatibility

With `radius=0` on all bodies, no obstacles, and `spin`/tangential force unused, this design reduces exactly to today's `CentralForceDynamics` gravity model — `V(d, 0, s_ij)` should collapse to the current `f(d)` gravity tail (verifiable by loading `gravity_central_v1.pt`'s weights into the radius-0 slice of the new potential and checking parity, or by retraining and checking the learned potential matches within the existing force-vs-analytic probe used for `gravity_relativistic_central.pt`, 0.56x-analytic-style comparison). This is a concrete zero-shot/parity test, not just a design intention.

## 6. Training data

**Resolved 2026-09-28: truth stays exclusively bounce.py's own penalty-force physics — no new/different truth engine.** `bounce.py`'s existing contact model (`penalty_force`, C1-smooth, zero and zero-slope at zero penetration, symplectic-Euler integration) already has exactly the right shape for this: `wall_force` treats the box boundary as four immovable line segments that push back on a ball via `penalty_force(radius - distance-to-wall, stiffness)`, and `ball_pair_forces` does the identical thing between two movable balls along their center line. Phase 1's two obstacle primitives are extensions of the *same* function family, not new physics:

- **Circle obstacle**: identical to `ball_pair_forces`' math (penalty force along the center line as a function of `2r - d`) between a real ball and one designated-immovable point — i.e. exactly a `kinematic=True` "ball" in bounce.py's own terms. This requires no new force function at all, only a way to mark one entry in the balls list as excluded from `integrate`.
- **Flat wall, placed anywhere (not just the box boundary), axis-aligned only (resolved 2026-09-28):** generalizes `wall_force`'s box-edge case (point-to-axis-aligned-line distance at a fixed location) to a movable axis-aligned segment at an arbitrary position — same horizontal/vertical orientation `wall_force` already handles, just no longer pinned to the four box edges. This is the piece that needs new code in `bounce.py` (today's `wall_force` special-cases the four fixed box edges at `0`/`n-1`; it needs a segment position and finite extent instead), but stays within `wall_force`'s existing axis-aligned math — no arbitrary-angle point-to-line-segment geometry, no polygons.

Both primitives get added to `bounce.py` (and kept as the CPU reference/test implementation, per the precedent already set for this codebase in `docs/debugging/z-and-mass-channels-feasibility.md` §truth sim/GPU generators) and to its existing GPU-vectorized counterpart (`scripts/ball_1k_rollout.py`-style torch port), which is what actually generates training data at scale — the standing GPU-only-compute rule still applies to *generation volume*, it's `bounce.py`'s physics that stays fixed as the one source of truth, checked bit-comparable between the CPU and GPU implementations the same way the existing physics already is.

Data-generation scope for Phase 1:
- Ball-ball contact with varying `(mass, radius)` pairs — extends the existing multi-mass truth-generator gap already identified in `docs/debugging/z-and-mass-channels-feasibility.md` (bounce.py / `ball_1k_rollout.py` truth is unit-mass only today). Mass ratio range: reuse the existing recommendation (0.25-4 train, 0.1/10 held-out test).
- Ball-vs-circle-obstacle scenes: circle position, radius, and count varied per scene.
- Ball-vs-wall-segment scenes: axis-aligned segment position and length varied per scene, orientation fixed to horizontal/vertical (plus the existing box-boundary walls, which are the `radius = box-scale` degenerate case of the same mechanism and need no separate handling).
- Each obstacle scene is rendered to the model as boundary-point samples (Section 4) at **swept** density (recommended default, since a fixed density is exactly the failure mode the boundary-sampling question describes) — this is a model-input-side transform applied on top of bounce.py's exact truth, not a change to how truth itself is computed.
- Spin/friction truth is out of scope for Phase 1 entirely (Section 0) — `bounce.py` has no tangential contact force or torque today, and none is added until Phase 2.

## 7. Testing plan

- **Parity**: radius-0/no-obstacle limit reproduces existing `CentralForceDynamics` gravity behavior (Section 5).
- **Conservation**: body-body-only (no obstacles) scenes conserve momentum and energy exactly, same test pattern as `tests/test_token_free.py::test_conservative_pair_momentum_exact` and `test_central_force.py`.
- **Generalization**: unseen obstacle shapes, positions/orientations, and boundary-point densities at inference (the actual target capability — "drop a circle in after training"); unseen mass/radius ratios; held-out seeds per [[feedback_select_on_heldout_seeds]]; real-retrain validation of any synthetic probe per [[feedback_validate_synthetic_loss_probes_with_real_retrain]].
- **Scaling**: O(N) and tiling preserved with obstacles present — boundary point count should not break the cell-list assumptions in `build_radius_graph_cells`.
- **Post-run video review**: fresh unbiased subagent review of rollout video once trained, per [[feedback_auto_run_video_analysis_on_sim_finish]].
- **Phase 2 (deferred)**: once spin/tangential is built, add a Newton's-third-law-with-friction test (momentum exact, energy strictly non-increasing from friction) and a spin-aware diagnostic render (e.g. a rotating tick mark per ball) — neither is needed or buildable for Phase 1, which has no spin channel.

## 8. Risks / open items

- Tangential force is new physics with no prior baseline in this codebase (unlike the normal/contact term, which is a generalization of existing `cons_pure` and `CentralForceDynamics` structure) — higher implementation and training risk than the rest of this design.
- Boundary-point sampling density is a real tunable tradeoff (fidelity vs. cost), not free generalization; needs its own probe (analogous to the existing neighbor-radius-vs-displacement issue) before committing to a default density.
- Combining gravity's long-range smooth tail and contact's steep short-range repulsion in one learned `V` is more capacity/harder to fit than either alone; the current gravity model and `cons_pure`'s contact model were each fit to only one regime. Should be checked as its own ablation (one potential fit to a mixed gravity+contact dataset) before assuming it just works.
- Spin as a first physical DOF is new; no existing truth sim, loss, or eval currently touches angular quantities at all.

## 9. Decisions (resolved 2026-09-28)

- **Obstacle primitives**: circles and flat walls, placed at arbitrary position; walls are axis-aligned only (no arbitrary angle), and polygons/corners are not in Phase 1 scope.
- **Truth engine**: bounce.py's own penalty-force physics only, extended in place (Section 6) — no separate/new truth simulator.
- **Boundary point sampling density**: swept as a training-time nuisance variable by default (Section 4/6), with the actual density values left as an implementation-time probe rather than fixed in this spec.
- **Phasing**: normal-contact term first (Phase 1, this spec's main scope); spin/tangential/friction deferred to Phase 2 and kept only as a documented future goal (Section 3), not built now.
