# Augur tiled scatter-field model in the browser

Static page: the scatter-field model run as a tiling, next to the exact simulation, as a 3D model view in the
style of bbycroft.net/llm (free orbit, layout standing on a wall). The weights are Exp E (`E_ms_kp_pot_v_g128`,
60769 iterations, trained on 10-200 bodies in a 64x64 box) plus the R1b far kernel; the tiling and the far kernel
are the ones in `scripts/scatter_tiling.py` and `scripts/scatter_far.py` (docs `scatter-tiling.md`,
`scatter-far-kernel-fix.md`). No retraining: every tile runs the same weights.

## What runs

An `nt x nt` grid of tiles (2x2 = domain 64, 4x4 = domain 128), each owning a 32-unit square and running one copy of
the model on its square plus a 16-unit halo, i.e. exactly the trained 128x128 grid (cell 0.5). Per force evaluation:

- ghost bodies: bodies of neighbouring tiles inside a tile's extended box are CIC-scattered into it and enter its pair
  term; only a tile's own bodies keep their force (bodies move between tiles by position every evaluation)
- halo field exchange: after each evaluation the halo cells of each tile's recurrent potential are overwritten by the
  neighbours' interior values (zeros outside the domain)
- far term: a coarse global grid (cell 2) carries the learned-kernel convolution of all mass; each tile adds the coarse
  potential minus its own box's coarse potential (mass outside its box) to the kernel acceleration that feeds the U-Net
  and the output
- global momentum re-centring; velocity Verlet with one evaluation per tick

The R1b kernel is the 4,417-parameter kernel MLP retrained (everything else frozen) on the exact softened law out to
r = 1500; the shipped kernel is repulsive beyond r = 64 (the kernel chart in the page shows both). The learned
ghost-closure edge fix is not part of this page.

Exact (no parameters): CIC scatter and gather, FFT convolution, finite-difference gradient, coarse avg-pool and bilinear
upsample, ghost and halo bookkeeping, momentum re-centring, Verlet. Learned: U-Net, kernel K(r), short-range pair MLP.
The exact simulation is softened gravity (eps 0.5, G 1, all pairs, 4 leapfrog substeps per tick, `../scatter/js/truth.js`).

## Page

Panels (wall layout): exact sim and model side by side, in lockstep (same tick from two buffers; the model panel shows
the tiles' potentials joined at the cores, ghost rings tinted by the receiving tile, the sim's body positions as faint
orange ghosts and a grey error link per body); the tile array (each tile a stack, front to back: potential, far term,
local-kernel potential, density; core square in the tile's colour, the rest of the slab is halo; gold arrows = ghost
bodies, thicker with more bodies; cyan = halo potential exchange); the far-term lane (coarse density, kernel force
curve, coarse potential, orange arcs to every tile); one tile's U-Net with skip connections (click a tile to inspect it).

Controls: space pause, `.` step, `r` reset, `g` / `x` / `f` toggle ghost bodies / halo potential / far term, `e` error
links, `a` acceleration arrows, `o` sim overlay on the model panel, `1`-`5` isolate one tile layer, `[` `]` channel
(-1 overlays all 32), `d` statistics (error vs sim, seam and interior error, ghost counts). Toggling a mechanism
restarts the model from the displayed tick with that mechanism changed (the sim is unaffected): the state restored is the
tick's accelerations and the tiles' potentials (16-bit companded, 3e-5 of the tile max). The sim and the model share the
tick; the model runs one tile per worker (hardware concurrency minus 2, at most the number of tiles) and a frame
buffer of 50 (2x2) or 24 (4x4) ticks ahead and 20 behind. Cost per tick: about 1 s (2x2) and 1.5-3 s (4x4) of tile
work spread over the workers (Node 24 and Chromium on 16 threads); the buffer fills while paused.

Hover: tile cell (layer, sim coordinates, core or halo), model potential, body (owner tile, model and sim state), U-Net
channel value. 4x4 holds up to 600 bodies, 2x2 up to 300. Bodies outside the domain get no grid force, only the pair term.

## Verify against PyTorch

    PYTHONPATH=<repo with scripts/scatter_tiling.py and scripts/scatter_far.py>:. python3 scripts/export_web_testvectors_scatter_tiled.py \
        --checkpoint E_ms_kp_pot_v_g128.pt --kernel results/far_kernels/r1b_mlp.pt --device cuda
    node web/scatter-tiled/tests/verify.mjs

Float64 PyTorch (`TiledScatter`, ghosts + field exchange + far, R1b) against the JS port, 4 scenes (2x2 and 4x4, 40-60
bodies, bodies crossing seams), 20 steps. Measured: 1-step max abs diff (positions and velocities) 3.8e-7, step-20 diff
3.3e-6, single-force acceleration diff at most 7.9e-6 for all five mechanism settings (full, no ghosts, no far, no field
exchange, none), tile potential diff 1e-5. JS tiled vs JS single whole-domain grid (G = 128 / 256, same weights and
kernel) after 20 steps: 1.4e-3 (2x2), 1.9e-3 and 2.5e-3 (4x4), equal to PyTorch's tiled-vs-single difference
(1.4e-3 .. 2.5e-3): the tiling reproduces the single grid up to the usual chaotic amplification of float differences.
`tests/frames.mjs` checks quantisation round trips and scene generation; `tests/ui.py <outdir>` drives the page with
Playwright (serve `web/` on port 8765 first): lockstep, mechanism toggles, tile inspection, hover, 4x4 switch.

## Shipped ticks and compute more

The default scene (4x4 tiles, cold collapse, 300 bodies, fixed seed 4738, all mechanisms on) ships ticks 0-100 of the model and of the exact
sim in `precomputed/`: `tiled4_collapse300.bin.gz` (planar binary, per-tick float32 bodies and accelerations, ghost lists, 8-bit
sqrt-companded tile fields with a per-tile max; density and potential at 128x128, local-kernel potential and far term at 32x32, no U-Net
activations), `tiled4_collapse300.json` (layout, kernel chart, weights summary) and `state100.bin.gz` (float64 bodies of the model and
the sim at tick 100, float32 potential windows of the 16 tiles). The page plays the 101 ticks from the file with no model code running: no
weights, tile worker or sim worker is fetched or started (checked in `tests/ui_precomputed.py`). "Compute more ticks" (a panel at tick 100, and a
nav button) loads `state100.bin.gz` and the weights, starts the tile workers, restores the model from the exact tick-100 state
(`TiledDriver.restore`) and the exact sim from its tick-100 state, and appends 20 / 50 / 100 ticks; Stop halts it and the same button resumes.
Computed ticks keep full-resolution fields and the U-Net; frames older than 20 ticks are compacted to the shipped format. Any other scene,
body count, grid, random seed or a mechanism toggle is computed live from tick 0 as before (a toggle at a shipped tick restarts from that tick's
compressed state, which is slightly off the exact state: see the last line of the test). `?live` skips the shipped ticks.

The panel states the horizon plainly: tick 100 is the longest horizon the project's rollout tests score (err@100), the model is not energy
conserving, the dynamics are chaotic and its error against the sim keeps growing; the tiled JS is checked against PyTorch for 20 ticks only.

Regenerate (about 5 min on 16 threads, writes the files above and `tests/ref_continue.json`, the continuous run's ticks 100-120):

    node web/scatter-tiled/tools/precompute.mjs 100 20

Check (`PARITY_TICKS=100` recomputes the whole shipped run from tick 0, about 4 min):

    node web/scatter-tiled/tests/precomputed.mjs
    python3 web/scatter-tiled/tests/ui_precomputed.py <outdir>     # serve web/ on port 8765 first

## Re-export weights

    PYTHONPATH=. python3 scripts/export_web_weights_scatter_tiled.py --checkpoint E_ms_kp_pot_v_g128.pt --kernel results/far_kernels/r1b_mlp.pt

## Files

- `js/tiled_model.js`: `TileNet` (R1b kernel on any grid size, coarse convolution, own-query pair term), `TileSet` (the
  tiles one worker holds: scatter phase, finish phase), `TiledDriver` (packing, ghosts, far term, halo exchange, momentum,
  Verlet, restore), kernel curves, scenes. Builds on `../scatter/js/scatter_model.js` (U-Net, FFT, MLP).
- `js/tiled_worker.js`: producer; spawns `js/tile_worker.js` workers, runs the Verlet loop and posts frames.
- `js/sim_worker.js`: the exact simulation, one message per tick.
- `js/tframes.js`: frame packing (16-bit per-tile fields, 8-bit U-Net of the inspected tile; compact 8-bit frames).
- `js/precomputed.js`: shipped-tick file format (encode, decode) and the tick-100 state.
- `tools/precompute.mjs`: builds `precomputed/` with the JS model on worker threads.
- `js/view3d.js`, `js/app.js`: the 3D view, playback, controls. Uses `../scatter/js/{stage,colors,plots,truth}.js`.
- `weights.bin` / `weights.json`: 264,836 fp32 weights (Exp E 260,419 + far kernel 4,417) and config.
- `tests/vectors.json`: PyTorch references.

## WebGPU U-Net prototype (branch web-scatter-tiled-webgpu)

Opt-in, not wired into the page. Only the tile U-Net forward runs as WebGPU compute shaders, batched over tiles; everything else and the CPU path are unchanged. Status 2026-10-23: numerics match on a software adapter (U-Net 1.2e-6 of peak vs the JS path, 20-tick rollout 5.7e-6 vs the JS CPU path, 3.75e-6 vs PyTorch float64); the speed criterion (>= 5x over CPU workers for 16 tiles) is UNTESTED because this machine has no hardware WebGPU adapter. Details in `docs/debugging/experiment-log.md`.

- `js/unet_gpu.js`: `UNetGPU.tryCreate(manifest, buf, T)` returns null without WebGPU (caller keeps the CPU path); `run(x)` takes float32 `[T][6][128][128]`, returns `[T][128][128]`.
- `js/tiled_gpu.js`: `GpuPool`, a drop-in for `LocalPool` that sends each finish phase's U-Nets to the GPU as one batch.
- `webgpu_bench.html` (serve `web/`, open `scatter-tiled/webgpu_bench.html?tiles=16` in a WebGPU browser): on-page GPU vs CPU-worker timing, GPU vs CPU numerics, PASS/FAIL line.
- `tests/webgpu_verify.mjs`: Node check (`npm i webgpu` somewhere, then `STEPS=20 WEBGPU_MODULE=<dir>/node_modules/webgpu/index.js node tests/webgpu_verify.mjs`).
