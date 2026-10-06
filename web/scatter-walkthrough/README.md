# Bounce scatter-field walkthrough

Static page: a 3D, step-through view of one traced tick of the scatter-field model, in the spirit of
bbycroft.net/llm. Fields (128x128 grids, 32-channel UNet slabs, potentials) are textured planes laid out
as a U (encoder along the front row, decoder back along the second); bodies are spheres above the arena.
Nine phases (overview, scatter, far field, UNet, potential and gradient, gather, pair term, integrate,
recurrence) are time-pure functions of the local time `t` in `js/walk.js`: camera keyframes, highlights,
labels and commentary are recomputed every frame, so scrubbing is free. `breakAfter`-style pauses stop
playback until Space / Next.

The model is the one in `../scatter/` (weights and `scatter_model.js` are reused by relative path). Passing
a `trace` object to `ScatterNet.force()` / `step()` records the intermediate tensors (input stack, phiK,
UNet encoder/decoder activations, phi, gradients, gathered and pair accelerations); it is off by default and
does not change results. `js/worker.js` answers `{cmd: "reset" | "trace"}` with `{type: "ready" | "trace" | "error"}`.
`step()` traces its second (end-of-tick) force pass; Verlet runs the pipeline twice per tick.

Controls: space play/pause, arrows prev/next, `.` next tick, `r` reset, `[` `]` channel, `0` re-apply the
scripted camera. Click a body to follow it; hover cells and bodies for values and linked highlights.

Tests: `node tests/verify.mjs` (traced == untraced bit-identical, trace shapes, self-consistency, phi vs the
PyTorch field in `../scatter/tests/vectors.json`); `python3 tests/ui.py <outdir>` and `tests/interact.py <outdir>`
drive the page with Playwright (serve `web/` on port 8765 first). UNet intermediate levels are not checked
against PyTorch (no reference dump exists).
