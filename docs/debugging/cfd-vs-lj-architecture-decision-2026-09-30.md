# CFD vs LJ: which foundation to converge on (2026-09-30)

Context: deciding whether to keep investing in the CFD contact-gen line
(`model/contact_force.py`) or the LJ line (`model/lj_force.py`) as the base for
a unified particle-interaction architecture. All other work (renders, Polaris
runs) paused until this is resolved.

User's stated goals, established this session:
1. **Force/interaction generality is the primary axis** (not scene/geometry
   generality) -- what kinds of physical interactions the architecture can
   represent, not what shapes of scene it can handle.
2. **Channel-energy compatibility matters now**, not as a deferred concern.
   Interactions may be non-conservative *per channel* (e.g. vx channel -> heat
   channel -> light channel), but total energy across the whole system must be
   conserved. This is the standard physics distinction between a
   *non-conservative force* (loses energy to an untracked channel, e.g.
   friction -> heat) and *violating the first law* (never valid for a closed
   system) -- the user's chain of thought here is correct, refined: the
   constraint belongs at the system-energy-accounting level, not baked into
   each pairwise force as a scalar-potential gradient.
3. **CFD and LJ are meant to converge into one architecture eventually.**
   Today's choice is which foundation to build that from, not a permanent
   fork.

## Core finding: both architectures share the same disqualifying ceiling

CFD's force is `-dV/dd` of a learned scalar potential `V(pen, r_sum, m_sum,
m_prod, m_diff)`, via autograd, constrained to act along the pair line
(`model/contact_force.py::ContactForceDynamics.accel`). LJ's force is a
learned radial function `f(d)*d^-7` along the minimum-image pair line
(`model/lj_force.py::LJForceDynamics.pair_force`). These look different
(potential-gradient vs. direct radial function) but are the same class: any
force that is purely a function of separation `d` and acts along the
inter-body line is automatically conservative (a vector-calculus fact, not a
training outcome). Neither model takes relative *velocity* as an input
feature at all, so neither can represent friction/drag/spin-coupling
regardless of training data. Picking between CFD and LJ does not, by itself,
get to the stated goal -- both are structurally capped at zero on the axis
that matters most.

## Where CFD and LJ actually differ

| | CFD (`ContactForceDynamics`) | LJ (`LJForceDynamics`) |
|---|---|---|
| Per-body attributes | mass, radius as pair features (`m_sum`, `m_prod`, `m_diff`) | none -- homogeneous units (eps=sigma=m=1 fixed), single species only |
| Scene geometry | kinematic obstacles injected as boundary points -- arbitrary shape at inference, proven on unseen circles | no obstacle/boundary concept at all; periodic box only |
| Demonstrated generalization | unseen obstacle shape + unseen wall density (`symlog_densershape`), both recover to/below `v2` baseline | density/temperature tiling to N=172032, held-out rho band (0.62-0.82) matches truth |
| O(N) scaling | uses `build_radius_graph_cells` (same primitive as LJ), not separately stress-tested at scale for this model | proven end-to-end: correctness 1e-15 vs dense, wall-clock genuinely O(N) to 172k, tiling-invariant (T, psi6 flat across 1024x range) |
| Known open failure | mass-ratio OOD sign-crossing in the symlog head -- root-caused, not fixed ([[project_cfd_contact_gen_variant_status]]) | near-field core (d<1.2) force law wrong by ~1000x -- root-caused (training-data coverage gap, not capacity), not fixed |
| Integrator | semi-implicit half-step (`dp` then `dv`) | full velocity-Verlet substeps |

CFD is ahead on the infrastructure a unified architecture actually needs --
heterogeneous per-body attributes and arbitrary injected geometry -- which
matters if convergence is the real goal, since LJ has neither and would need
both rebuilt from scratch. LJ is ahead on proof that the O(N) neighbor search
(`model/token_graph.py::build_radius_graph_periodic` /
`build_radius_graph_cells`) scales cleanly at real particle counts with real
physics on top, which the CFD model itself hasn't been stress-tested at (only
a sanity check on edge count in `scripts/eval_contact_generalization.py`, not
a full scaling+tiling run like LJ got in `scripts/lj_scaling.py`).

## Recommendation

Not "continue CFD" or "continue LJ" as a binary. CFD's pair-feature /
obstacle-injection infrastructure is the right base to converge toward, but
its force *head* needs to be replaced -- it cannot represent what the user
wants no matter how it's retrained. LJ's O(N) cell-list work is already a
drop-in for whichever survives, since both import the same
`model/token_graph.py` primitives.

Concrete architectural change: keep CFD's pair-feature setup (mass, radius,
kinematic obstacles) and **antisymmetrize the force output directly**
(`F_ij = -F_ji`, guaranteeing Newton's third law / momentum conservation by
construction) **without** routing it through a scalar potential or
constraining it to the pair-line direction, and **add relative velocity as an
input feature** so friction-like/non-central terms are representable. Global
energy conservation then becomes a system-level accounting check (sum of
channel energies across a step, checked/regularized, not architecturally
forced per pair) -- matching the "conserved overall, not per-channel"
framing above.

This is a real scoped design change (new force-head architecture, new loss
terms for energy accounting), not a tweak to an existing CFD/LJ variant. Not
yet written up as a full spec or built.

## Status

**Paused 2026-09-30, on the user's explicit instruction: no further work,
including renders or additional Polaris runs, until this architecture
question is resolved.** Next step, if/when resumed: either write this up as
a formal design spec (same process as
`docs/superpowers/specs/2026-09-28-cfd-contact-generalization-design.md`) or
continue discussing the antisymmetric-force-head idea further first -- user's
call, not yet decided.

See also [[project_cfd_contact_generalization_spec]],
[[project_cfd_contact_gen_variant_status]], [[project_core_requirements]].

## Follow-up session, same day: refining the antisymmetric-force-head design

Resumed from where the pause left off. Walked `ContactForceDynamics` end to
end again (neighbor finding via `build_radius_graph_cells` -> 5-feature
`pair_features` (`pen, r_sum, m_sum, m_prod, m_diff`) -> gated potential MLP ->
`-dV/dd` autograd -> Verlet half-step), then worked through what changes when
the potential-gradient force is replaced with a directly antisymmetrized one.
Net effect: the redesign direction from the first session narrowed from a
general idea into a concrete edge-level implementation plan, plus two
corrections to proposals raised along the way.

**Mass-feature choice, resolved.** `m_sum`/`m_prod`/`m_diff` are swap-symmetric
only because the current potential has to produce the *same* scalar `V`
regardless of edge direction -- antisymmetry then falls out for free from
`F = -del V` flipping sign with `rel`'s direction. Once the force is output
directly instead of derived from a potential, that symmetric-feature trick is
no longer required *or* desirable: it can't distinguish "I am the heavy body"
from "I am the light body," which matters once asymmetric effects
(velocity-dependent drag, directional friction) are in scope. Decision:
features become directional (`mass_src`, `mass_dst`, not symmetrized), with
antisymmetry enforced architecturally instead (see below). `pen` and `r_sum`
must both still be fed as separate scalars (`pen` alone collapses away the
absolute contact scale -- 0.1 penetration means something different for
radius-0.4 vs radius-5 bodies); `rel_vel` (full 2D vector, not a radial
projection) is the new feature needed for drag/friction. Target feature set:
`(pen, r_sum, mass_src, mass_dst, rel_vel)`.

Also surfaced, not yet decided: obstacle points are hardcoded `mass=1.0` and
rely entirely on the `kinematic` flag (not the mass value) to behave as
immovable -- once mass becomes a directional feature the network actually
conditions on, does an obstacle need a real (large) mass value, or does the
placeholder + kinematic-mask split still hold? Separate axis from the
ball-ball `MASS_RANGE` fix already in flight (see below), not yet addressed.

**Complexity check: direct force output does not add an O(k^2) cost, and is
actually a chance to remove 2x redundant work that exists today.** The
worst-case density of the neighbor graph (dense local clusters producing many
pairs) is a property of `build_radius_graph_cells`'s radius cutoff, identical
regardless of what the per-edge model does with the resulting edges --
nothing about swapping the force head changes that. What *does* change:
today's symmetric `feat` makes the potential MLP compute the identical scalar
`v` on both directed copies of every undirected pair (`E = 2P`), purely to
get the antisymmetric force sign for free from the gradient. With directional
features, antisymmetry has to be enforced structurally instead: dedupe edges
to one canonical direction per undirected pair, run the network once per pair
to get a raw force, then `index_add` it as `+F` onto one body and `-F` onto
the other. That's `P` network evaluations instead of `2P` -- half of today's
compute, not more.

**Correction: a 4-output split (`src_sign, src_mag, dst_sign, dst_mag`) would
throw away momentum conservation by construction, which is the entire point
of antisymmetrizing.** Two independently-parameterized per-body outputs have
nothing forcing `F_src = -F_dst`. Unlike energy -- where "conservative per
channel" is legitimately too strong a constraint, since friction genuinely
moves energy into heat -- Newton's third law for a two-body pairwise
interaction holds essentially universally in classical mechanics; friction
between two contacting bodies is still an equal-and-opposite force pair, it
just isn't conservative in the energy sense. So the "weaken this constraint"
move that's correct for energy does not transfer to momentum. Resolution:
keep a single raw output per pair (optionally decomposed into normal +
tangential components via sign+magnitude, so up to 4 numbers *describing one
force vector*, not 4 independent per-body outputs), applied with `+`/`-` to
the two bodies via the same dedup/scatter step above. Normal direction stays
`rel/d`; tangential is the in-plane perpendicular, with `rel_vel` needed to
fix which perpendicular sign is physically "forward" (a magnitude alone has
no direction basis without it).

**Open and unresolved: multi-channel force output ("different mass collisions
for free").** Proposed outputting `(sign, mag)` pairs per channel, letting
forces flow across channels. Flagged as conflating two different things,
not yet resolved: (a) if "channel" means the heat/light energy-type channels
from the first session, those are scalar per-body accumulators with their own
update rule, not vector quantities exchanged antisymmetrically between
bodies (friction raises *both* contacting surfaces' heat; it doesn't move
heat off one body's ledger onto the other's) -- the antisymmetrized-force
machinery doesn't apply to them at all, so this doesn't get channel-energy
transfer "for free." (b) if "channel" instead means additively splitting the
single mechanical force into multiple component heads (e.g. normal/tangential
sign-mag heads summed before antisymmetrizing), that's architecturally fine
but doesn't obviously add new mass-dependence expressivity beyond what a
single-channel network already has, since `mass_src`/`mass_dst` are already
directional inputs. Needs the user to clarify which is meant before this
branch can be designed further.

## Active, separate from this pause

A targeted fix to the mass-ratio OOD blind spot on the *already-adopted*
`symlog_densershape` checkpoint (widen `MASS_RANGE`, retrain, re-eval -- see
[[project_mass_range_extension_active]]) was explicitly authorized by the
user as parallel work and dispatched to a subagent. It does not reopen or
resolve this pause -- it is a data-range fix to the current architecture, not
the redesign discussed above.

## Status (updated)

Still **not resolved** -- no formal spec written, no code changed on this
branch for the redesign itself. The antisymmetric-force-head direction is
now concrete at the edge/feature level (directional features, dedup edges,
single antisymmetrized output per pair, normal+tangential decomposition) but
the multi-channel question above is open, and the obstacle-mass question is
unaddressed. Next step unchanged: either write this up as a formal design
spec or keep discussing -- user's call.
