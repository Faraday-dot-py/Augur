# Bounce scatter-field model in the browser

Static page: the learned scatter-field N-body model (`B_ms_kp_pot_v_g128`, trained on 10-100
unit-mass bodies, 8087 iterations in a fixed 900 s budget, err@5/10/20 = .0006/.0014/.0039,
err@100 = .374 vs CentralForce .0006/.0011/.0025, .278; see
`docs/debugging/scatter-field-fixed-budget.md`) runs in plain JavaScript. The page is a single 3D view of
the model's structure (input stack, kernel branch, UNet encoder/decoder with skips, potential, gradient,
grid acceleration, recurrence, weights) with the current simulation's tensors drawn on it and the bodies
moving over the input stack. An exact softened-gravity ground truth (orange ghost bodies, error and energy
comparison) is opt-in from the dev menu.

One tick: CIC-scatter body density and momentum onto a 128x128 grid (cell 0.5, arena
64x64), FFT-convolve the density with a learned radial kernel (far field), run a 5-level
UNet on [scatter, kernel acceleration, previous potential] to get a potential, take its
negative gradient, bilinearly gather at each body, add a learned short-range pair term
(r < 2, 16 nearest), re-centre momentum, and advance with velocity Verlet (one force
evaluation per tick). The UNet is ~95% of the cost: ~0.4 s per tick (Node 24), so the model
runs in a Web Worker and the page shows whatever tick rate the machine sustains. Bodies
outside the 64x64 arena get no grid force (not trained there).

Controls: space pause, `.` step, `r` reset, `d` dev menu, `[` / `]` or the channel slider scrub the 32
UNet channels. Drag orbits, wheel / pinch zooms, hover (or tap) shows names and raw values; hovering a UNet
slab shows that layer's conv weights. Bodies move every tick; the slabs refresh from a traced force pass
whenever the worker is free (latest tick wins), so while playing they lag the bodies and the key shows
"slabs: tick k" with "updating" while behind. The bottom bar fades to 25% opacity when the pointer leaves
it (on touch it shows for 3 s after any tap).

The dev menu (`d`) holds the stats, the energy-drift plot of the model (energy is only computed
while it is open, baseline = first value after it opens or the model resets), a "Ground truth
(slow)" toggle, the error-vs-truth plot (shown while truth is on) and state export. Enabling
truth starts it from the model's current state ("truth since tick N") in its own worker, which
catches up to the latest tick and reports its lag; disabling terminates the worker.

By default the page plays precomputed runs (Precomputed button toggles Live): seed 4738, N = 10..100 in
steps of 10, 100 ticks each, played at 10 ticks/s with a tick scrubber (the live page starts from a random
state. Positions and velocities are exact outputs of this model; the slabs recompute the tensors on demand
from the stored state, with the recurrent previous potential warmed from the two preceding ticks (about 5-9%
relative L2 off the live field). Regenerate with
`node tools/precompute.mjs` (about 35 s per run on one CPU; the shipped files were generated on Polaris and
`node tests/precomputed.mjs` checks they regenerate bit-exactly).

Open `index.html` through any static server (`python3 -m http.server` from the `web/`
directory, then visit `/scatter/`). The landing page at the Pages root links here; this page has no link back.

## Files

- `js/scatter_model.js`: port of `ScatterField` (`scripts/scatter_field.py`): scatter/gather,
  FFT kernel convolution, UNet (4-output-channel blocked 3x3 convs), pair term, Verlet.
- `js/worker.js`: steps the model off the main thread and posts positions and velocities
  (transferred) and the energy only while the dev menu is open; answers `trace` (live state) and
  `tracetick` (stored state + warm ticks) with one traced force pass that leaves the sim untouched.
- `js/truth.js`: softened all-pairs gravity (symmetric pair loop), leapfrog, energy.
- `js/truth_worker.js`: runs the ground truth off the main thread, created only when enabled.
- `js/model3d.js`: the page's 3D view: the model's layout drawn with the current sim's tensors, free
  orbit and hover values, no walkthrough. Reuses `../scatter-walkthrough/js/{stage,layout,colors,cic}.js`.
- `js/trace.js`: transfer-list helper shared by this worker and the walkthrough worker.
- `js/plots.js`, `js/app.js`: dev-menu plots, playback, trace scheduling, controls.
- `weights.bin` / `weights.json`: fp32 weights + manifest/config (260419 floats).

## Re-export weights

    PYTHONPATH=.:<repo with scripts/scatter_field.py> python3 scripts/export_web_weights_scatter.py --checkpoint checkpoints/budgetB/sfv/B_ms_kp_pot_v_g128.pt

`config` in `export_web_weights_scatter.py` must match the training variant
(`ms_kp_pot_v_g128`, Exp B: extent 64, in_scale 1).

## Verify against PyTorch

    PYTHONPATH=.:<repo with scripts/scatter_field.py> python3 scripts/export_web_testvectors_scatter.py
    node web/scatter/tests/verify.mjs

`verify.mjs` compares free float32 JS rollouts (8, 30, 100 bodies, 20 steps) with float64
PyTorch. Measured: 1-step max abs diff 2.3e-7, step-20 diff 2e-6, potential field 4e-7.
- `precomputed/`: `index.json` + `nNNN.bin` (32-byte header, then float32 `[pos(2N), vel(2N)]` for ticks 0..100), 868 KiB total.
- `tools/precompute.mjs`, `tests/precomputed.mjs`, `tests/precomputed.py`: generator, node check, Playwright check.
