# Augur central-force model in the browser

Static page: the learned central-force N-body model (`checkpoints/gravity_central_v1.pt`,
a single distance-only pairwise-force MLP, 4353 floats, err@20 = 0.0054 on held-out
100-1000 body scenes, generalizes past 10,000 bodies) runs a live gravity cluster
in plain JavaScript, with a softened-gravity ground-truth ghost overlay for comparison.

Unlike the token model, this one has no walls, no box, and no per-ball recurrent
state -- it was never trained with any of those, so none are ported (see
`docs/debugging/unified-dynamics-relativistic-speed-limit.md` section 3).

Open `index.html` through any static server (`python3 -m http.server` from the
`web/` directory, then visit `/gravity/`). Linked from the landing page at the
Pages root (`web/index.html`).

## Files

- `js/central_model.js`: port of `CentralForceDynamics.accel` + `.forward` (`model/central_force.py`):
  the pairwise force MLP, a grid-bucketed neighbour-radius graph, and velocity-Verlet integration.
  `step(pos, vel, count)` steps in place.
- `js/truth.js`: port of `scripts/gravity_sim.py` (softened all-pairs gravity, leapfrog), for the ghost overlay.
- `js/plots.js`: force-curve and energy canvas plots.
- `js/app.js`: UI wiring, canvas rendering, controls (space, `.`, r, g, e).
- `weights.bin` / `weights.json`: fp32 weights + manifest/config (4353 floats).

## Re-export weights

    PYTHONPATH=. python3 scripts/export_web_weights_central.py --checkpoint checkpoints/gravity_central_v1.pt

Config (`dt` 0.1, `neighbor_radius` 100.0) matches `CentralForceDynamics`'s
defaults, which is what `gravity_central_v1.pt` was trained and evaluated with
(`scripts/polaris_train_gravity_central.sh`).

## Verify against PyTorch

    PYTHONPATH=. python3 scripts/export_web_testvectors_central.py
    node web/gravity/tests/verify.mjs

`verify.mjs` gates 1-step max abs diff (< 1e-4) against a float64 PyTorch step
from the same float32 states, and reports free-run diffs against the float32
PyTorch rollout at step 20 and at the end of each scene (8-200 bodies, 40-60 steps).
Measured: 1-step diff ~1e-15 (no substep loop or wall logic to round differently,
so it tracks PyTorch far tighter than the token model's ~1.9e-6), free-run diff
up to ~9e-5 by step 40-60.
