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

## Model view

`Model` (key `m`) swaps the 2D cluster for a 3D diagram of one body's step, in the style of the token page
(drag orbits, wheel/pinch zooms, hover shows name and value, bars are colored/sized by value). Click a body in
the 2D view (or `Pick`, `p`) to choose it; in the model view, click a bar of a neighbour's column to follow that
body. The diagram is the live trace of the step just computed, from `CentralNet.step(..., traceIdx)`:

- input token: position and velocity of the body;
- one column per neighbour (the 5 strongest by |force|, out of every body within `neighbor_radius`): offset,
  distance d, MLP input ln d, all 64 + 64 tanh activations of the force MLP (1 -> 64 -> 64 -> 1), MLP output f,
  1/(d^2+1), the force magnitude f/(d^2+1) and its x/y components on the body;
- the sum over all neighbours, a0;
- velocity Verlet: dp = dt^2 a0 / 2, the predicted position pos + vel dt + dp (which is also the new position),
  the second force pass at the predicted positions (same MLP, same weights; its 5 strongest pair forces are drawn)
  giving a1, and dv = dt (a0 + a1) / 2;
- output token: new position, new velocity.

The side panel's neighbourhood plot shows the same body, its drawn neighbours and the force arrows; the pair-force
plot marks their distances and overlays the true softened-gravity law d / (d^2 + 0.25)^1.5 (dashed), which the learned
curve matches to about 1% for d from 0.3 to 100 (checked at 8 distances). three.js is loaded lazily from jsDelivr on first use of the model view.
`js/arch.js` subclasses `web/token/js/arch.js` (`Arch`, colour map); `js/view.js` is the scene/labels/picking.

## Files

- `js/central_model.js`: port of `CentralForceDynamics.accel` + `.forward` (`model/central_force.py`):
  the pairwise force MLP, a grid-bucketed neighbour-radius graph, and velocity-Verlet integration.
  `step(pos, vel, count)` steps in place.
- `js/truth.js`: port of `scripts/gravity_sim.py` (softened all-pairs gravity, leapfrog), for the ghost overlay.
- `js/plots.js`: force-curve (with true-law overlay), neighbourhood and energy canvas plots.
- `js/arch.js`, `js/view.js`, `model.css`: the 3D model view (see above).
- `js/app.js`: UI wiring, canvas rendering, controls (space, `.`, r, g, e).
- `weights.bin` / `weights.json`: fp32 weights + manifest/config (4353 floats).

## Re-export weights

    PYTHONPATH=. python3 scripts/export_web_weights_central.py --checkpoint checkpoints/gravity_central_v1.pt

Config (`dt` 0.1, `neighbor_radius` 100.0) matches `CentralForceDynamics`'s
defaults, which is what `gravity_central_v1.pt` was trained and evaluated with
(`scripts/polaris_train_gravity_central.sh`).

## Verify against PyTorch

    node web/gravity/tests/trace.test.mjs

`trace.test.mjs` checks the per-body trace (summed edge forces, both force passes, output state) against `step()` itself.


    PYTHONPATH=. python3 scripts/export_web_testvectors_central.py
    node web/gravity/tests/verify.mjs

`verify.mjs` gates 1-step max abs diff (< 1e-4) against a float64 PyTorch step
from the same float32 states, and reports free-run diffs against the float32
PyTorch rollout at step 20 and at the end of each scene (8-200 bodies, 40-60 steps).
Measured: 1-step diff ~1e-15 (no substep loop or wall logic to round differently,
so it tracks PyTorch far tighter than the token model's ~1.9e-6), free-run diff
up to ~9e-5 by step 40-60.
