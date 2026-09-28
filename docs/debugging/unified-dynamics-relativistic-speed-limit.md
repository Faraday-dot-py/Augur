# Unified dynamics + architectural speed limit — design conversation (2026-09-28)

Parked for later implementation. No code written yet against this doc.

## 1. Relativistic |v| < c should be architectural, not a loss term

Loss-based penalties for `|v| < c` are dataset-specific (tuned weight,
doesn't transfer to a sim without that bound). Fix: predict momentum `p`
instead of velocity, convert to velocity only via a fixed, non-learned
boundary map:

    v = c * p / sqrt(1 + |p|^2 / c^2)

This is the same convention `gravity_sim.py`'s truth generator already
uses (`rollout_torch(relativistic=True)`) and matches `orbit_bh.py
--relativistic`. Range is the open ball of radius `c` for any finite `p`
— exact by construction, holds under OOD extrapolation, doesn't touch
the loss function so it composes with anything else in the objective.
Squash the **norm**, not per-component (per-axis tanh can still yield
Euclidean norm > c).

Key point (see section 2): relativity doesn't add new physics to a
central force — `dp/dt = F(x)` is the same equation either way, only the
momentum->velocity map changes. So this fix requires no new learned
parameters if the force law is already momentum-conserving.

## 2. CentralForceDynamics scales, TokenFreeDynamics doesn't — why, and how to get both

- **CentralForceDynamics** (`model/central_force.py`): pairwise force
  `f(d)` learned as a pure function of distance, summed via `index_add`,
  antisymmetric ⇒ exact momentum conservation, velocity-Verlet (symplectic).
  Trained on 100-1000 bodies, holds to 12,800 bodies (err@20 0.0054 vs
  token model's 2.14 at the same scale — `docs/debugging/experiment-log.md`
  "N-body 10k escape" entry, job 2938/2939). Generalizes because it's
  *structurally* forced to treat every pair identically regardless of N.
- **TokenFreeDynamics** (`model/token_free.py`): GRU/mirrored recurrent
  core + wall_head + pair_head. Flexible, learns collision impulses and
  wall bounces from data, but the recurrent/mirrored core's input
  statistics shift with N — **leading hypothesis, not yet confirmed by
  ablation** for why it fails to generalize past training body-count.
- **`CentralForceDynamics`'s own relativistic incapacity is fixed for
  free by the same change as section 1**: its docstring notes
  `v_{t+1} = v_t + dt*a` has no v-dependence, "cannot represent
  saturation even in principle." Relabeling its integrated state from
  `v` to `p` (force law unchanged) fixes this with zero new parameters.

### Proposed unified architecture (not yet built)

    class UnifiedDynamics(nn.Module):
        long_range(positions)             -> pairwise f(log d), all-pairs or far-field mesh, no cutoff
        contact(positions, velocities)    -> pairwise f(d, rel_v, penetration), cutoff radius, index_add
        # no wall_head, no GRU/mirrored core (see section 3)
        p_new = p + dt * (long_range + contact)
        v = c * p / sqrt(1 + |p|^2/c^2)    # fixed boundary map, both terms

One class, one set of weights, one forward call. `long_range` carries
gravity/many-body (borrow `CentralForceDynamics.force` verbatim);
`contact` carries collisions (borrow the existing antisymmetric
`pair_head` verbatim — already `index_add`-summed, not
softmax-normalized, per its own docstring in `model/token_free.py`).
Both additive into one momentum state, one speed-limit map applied once
to the combined result. A scene with no walls (gravity) gets zero
contribution from `contact` outside its cutoff — no wall-token input
needed (see section 3), consistent with the fixed-box-is-in-the-weights
resolution below.

**Open / unverified**: whether `contact`'s MLP-on-invariants (uses
relative velocity, not just distance) generalizes across N as cleanly as
`long_range`'s pure-distance law. Proposed test before committing: ablate
the GRU/mirrored core out of the gravity path and rerun the existing
20-12,800 body scaling benchmark (`gravity_tiling_test.py` harness) to
confirm it's the actual cause of the token model's scaling gap.

## 3. wall_head doesn't generalize — resolved decision (found this session, from 2026-09-26 conversation, session `762042ce-df36-4f4c-b8b4-e99df3f25e66`)

User's question at the time: *"Why feed in the environment as box tokens,
isn't that information stored in block A's weights?"* Resolution: for a
**fixed** trained box, wall geometry needs no runtime input at all — it's
a learned parameter, not an observation. Box tokens / geometry-as-input
should only be reintroduced if a sim needs *variable* box geometry at
test time (domain generalization), and even then as an optional
extension, not built into v1.

Direct implication for `model/token_free.py`: `wall_head`,
`wall_features`, `wall_contact_features` hard-code absolute position
relative to a fixed box (`x, (n-1)-x, y, (n-1)-y`) — sim-specific,
doesn't exist for wall-less sims (gravity, LJ molecular dynamics). These
should be stripped from any shared/unified dynamics core; a bounce-in-box
deployment can keep a wall term as a separate additive piece, not a
universal component of the base architecture.

## Verification plan before calling any of this done

1. Ablate GRU/mirrored core from the gravity path, rerun scaling
   benchmark, confirm it's actually the cause of the N-scaling gap.
2. Rerun `train_gravity_relativistic.py` with the momentum-state model,
   confirm `frac_scenes_over_c` hits exactly 0 at every step, not just
   step 20.
3. Confirm momentum conservation / energy drift still match
   `CentralForceDynamics`'s existing numbers (<0.01% at step 20) after
   the p/v relabeling.
4. Strip `wall_head`/`wall_features`/`wall_contact_features` from the
   shared core; re-verify bounce-in-box benchmark with wall term moved to
   a separate additive piece.

## Status

Design only — nothing implemented. Best current checkpoints as of this
writing: token model = `results/cons_pure.pt` (conservative_contact,
see [[project_conservative_contact_model]]); central-force =
`checkpoints/gravity_central_v1.pt` (see
[[project_gravity_generality_test]]). Neither has the momentum-based
speed limit or the unified-model treatment yet.
