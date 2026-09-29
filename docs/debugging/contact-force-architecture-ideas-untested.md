# Contact-force architecture ideas from conversation — untested

Context: symlog output-head test (this branch, `worktree-cfd-contact-gen`, commit
`2d07230`) grew out of a longer design conversation about `model/contact_force.py`.
Only one of the ideas discussed got tested. This file tracks the rest so a fresh
session can pick up without re-deriving the conversation.

## Tested (for reference — don't redo)

`ContactForceDynamicsSymlog`: potential head = `tanh(sign_head(feat)) *
exp(log_scale_head(feat))`, same shared trunk, same pairwise features, still
differentiated via `-dV/dd`. Fixed the wall-pinning failure (err@30
7.28/9.58 -> 0.66/0.39) but regressed `unseen_obstacle_shape` generalization
(err@20/30 0.046/0.044 -> 0.33/0.33) and gave mixed mass-ratio results. Not
adopted as default; kept as `--model symlog`. See `docs/debugging/experiment-log.md`
and commit `2d07230` for full numbers.

## Untested ideas from the same conversation

### A. Direct local-patch delta kernel (the original proposal, before symlog)

Replace the pairwise-potential-gradient design entirely: a learned kernel
convolves a local patch of neighbor tokens around a particle and outputs a
state delta directly (dp, dv, and in principle any other per-token attribute
if all of a particle's state lives in its token) — applied manually, no
autograd-through-a-potential step. Discussed as keeping O(N) as long as patch
size / neighbor count stays bounded independent of N and of local density
(reuses the existing `build_radius_graph_cells` neighbor search).

Open problem carried over from that discussion: a direct delta output has no
structural momentum/energy conservation the way `-dV/dd` does. Two ways raised
to fix that were themselves not tested independently of the symlog head:

- **Explicit equal-opposite delta per edge**: since `build_radius_graph_cells`
  already returns both directed edges per pair, symmetrizing via `index_add`
  over both directions was noted as already-implicit in the current design —
  worth confirming this actually holds for a *direct* delta kernel (not a
  potential gradient) rather than assuming it carries over.
- **Two independent per-particle kernel evaluations, combined via
  abs-average + resign**: `o_i` computed from particle i's own local
  neighborhood/hidden state, `o_j` from j's, magnitude = `(|o_i|+|o_j|)/2`,
  sign taken from pair geometry (e.g. penetration direction), applied
  +/- along the pair axis. Flagged as conserving momentum by construction
  but *not* energy (no potential-gradient structure), and as only justified
  if each side genuinely carries different context beyond the pairwise
  features already in `pair_features()` — otherwise it's a redundant,
  more expensive version of the single shared-feature evaluation already
  in use. Not implemented or tested in either form.

### B. Grid/PIC-style force (rasterize -> convolve -> interpolate)

Deposit particles onto a fixed-resolution grid (CIC-style), run a local
convolution over the grid to get a force field, interpolate back to particle
positions — drops the radius-graph/neighbor-search entirely, true O(N) with
no log factor (vs. the current O(N log N) sort-based graph build).

Explicitly flagged as a different *physics* model, not a free reimplementation:
- Re-opens the momentum-conservation-via-matched-deposit/interpolate-kernel
  problem this project already hit once (windowed-attention era, "CIC+FFT
  mismatch" bug).
- Loses exact pairwise contact resolution (`pen = r_i + r_j - d` on a specific
  pair) in favor of a mean-field/continuum force — multiple particles sharing
  a grid cell/kernel footprint can't be individually resolved. Better fit for
  genuine continuum forces (pressure/viscosity) than hard-body contact, which
  is what `ContactForceDynamics` is actually for.
- Grid resolution needs to be finer than the smallest particle spacing to
  resolve contact geometry at all — reintroduces the density-dependent
  resolution problem noted in the (separate-worktree) dense-region-scaling
  notes as the reason a fixed-grid/LUT approach was rejected there.

Not started. If picked up, scope it as a CFD/continuum-force variant, not a
drop-in replacement for `ContactForceDynamics`.

### C. Discrete log-scale bucket magnitude head (the literal original proposal)

Before settling on the 2-neuron `tanh(sign) * exp(log_scale)` version, the
actual proposal on the table was: 1 sign neuron + 3-4 neurons "activating"
per log-scale magnitude bucket + a base-value neuron, i.e. a discrete
mixture-of-scales output (closer to distributional-RL-style binned regression
than to symlog). The 2-neuron version was tried first as the simpler,
continuous alternative (no bucket boundaries to place or interpolate between).
It was not compared against the bucket version. If the symlog head's
obstacle-shape regression turns out to be a smoothness/extrapolation problem
rather than a capacity problem, the bucket version (softmax-weighted mixture
over bins, kept differentiable — no hard argmax) is the next thing to try,
since a mixture-of-experts-style magnitude head could behave more
conservatively outside the training distribution than a single unconstrained
`exp(log_scale)`.

## Likely next step (not started)

Symlog fixed wall-contact by giving the head more dynamic range but paid for
it in obstacle-shape generalization — suggests the extra range is being used
to overfit under-sampled obstacle-shape/long-horizon cases. Two directions
raised but not tried: denser obstacle-shape sampling paired with the symlog
head, or capping/regularizing `log_scale`'s effective range so it can't
extrapolate as freely. Neither implemented yet.
