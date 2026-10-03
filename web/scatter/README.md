# Bounce scatter-field model in the browser

Static page: the learned scatter-field N-body model (`B_ms_kp_pot_v_g128`, trained on 10-100
unit-mass bodies, 8087 iterations in a fixed 900 s budget, err@5/10/20 = .0006/.0014/.0039,
err@100 = .374 vs CentralForce .0006/.0011/.0025, .278; see
`docs/debugging/scatter-field-fixed-budget.md`) runs live in plain JavaScript, with a
the model's learned potential drawn as a heatmap. An exact softened-gravity ground truth
(orange ghost, error and energy comparison) is opt-in from the dev menu.

One tick: CIC-scatter body density and momentum onto a 128x128 grid (cell 0.5, arena
64x64), FFT-convolve the density with a learned radial kernel (far field), run a 5-level
UNet on [scatter, kernel acceleration, previous potential] to get a potential, take its
negative gradient, bilinearly gather at each body, add a learned short-range pair term
(r < 2, 16 nearest), re-centre momentum, and advance with velocity Verlet (one force
evaluation per tick). The UNet is ~95% of the cost: ~0.4 s per tick (Node 24), so the model
runs in a Web Worker and the page shows whatever tick rate the machine sustains. Bodies
outside the 64x64 arena get no grid force (not trained there).

Controls: space pause, `.` step, `r` reset, `f` field, `d` dev menu. Mouse wheel / pinch zooms
about the cursor, drag pans (one finger on touch), double-click zooms 2x, `+`/`-` zoom, arrow
keys pan, `0` or the Fit button returns to auto-fit (any pan/zoom switches to manual). Bodies
are drawn 0.5 units wide but never smaller than 3.5 px. The bottom bar fades to 25% opacity when
the pointer leaves it (on touch it shows for 3 s after any tap).

The dev menu (`d`) holds the stats, the energy-drift plot of the model (energy is only computed
while it is open, baseline = first value after it opens or the model resets), a "Ground truth
(slow)" toggle, the error-vs-truth plot (shown while truth is on) and state export. Enabling
truth starts it from the model's current state ("truth since tick N") in its own worker, which
catches up to the latest tick and reports its lag; disabling terminates the worker.

Open `index.html` through any static server (`python3 -m http.server` from the `web/`
directory, then visit `/scatter/`). The landing page at the Pages root links here; this page has no link back.

## Files

- `js/scatter_model.js`: port of `ScatterField` (`scripts/scatter_field.py`): scatter/gather,
  FFT kernel convolution, UNet (4-output-channel blocked 3x3 convs), pair term, Verlet.
- `js/worker.js`: steps the model off the main thread and posts positions and velocities
  (transferred), the potential only while the Field overlay is on, and the energy only while the
  dev menu is open.
- `js/truth.js`: softened all-pairs gravity (symmetric pair loop), leapfrog, energy.
- `js/truth_worker.js`: runs the ground truth off the main thread, created only when enabled.
- `js/model3d.js`: the Model view (key `m`): the model's 3D layout drawn with the current sim's
  tensors, free orbit and hover values, no walkthrough. Lazy-loaded with three.js on first open; the
  worker answers a `trace` command with one force pass on a copy of the recurrent state, so the
  sim is unaffected. Reuses `../scatter-walkthrough/js/{stage,layout,colors,cic}.js`.
- `js/trace.js`: transfer-list helper shared by this worker and the walkthrough worker.
- `js/plots.js`, `js/app.js`: canvas plots, dirty-flag rendering, camera, controls.
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
