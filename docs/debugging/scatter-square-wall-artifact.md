# Square "walls" in the scatter-field BH rollouts

Question: why do the black-hole rollouts (videos/scatter_bh_1b_density.mp4, 100k grids) look like particles bounce off square walls? Date 2026-10-08. Model: Exp E `ms_kp_pot_v_g128` (grid 128, extent 64, so grid box [-32, 32]^2, cell 0.5). Scene: Gaussian sigma 8 clipped to +-29, vfac 0.3, c 10, dt 0.1, unit masses, seed 4738.

## Answer (one paragraph)

There are two separate things and neither is a wall that bounces particles in the way the video suggests. (1) The visible box, curtains and rounded square are mostly display and initial-condition geometry: the IC is a Gaussian truncated at +-29 in each axis (a square support, visible as a sharp edge on a log-density plot at N=1e9 even without any clipping pile-up), plus 601 per 1e6 bodies piled exactly on the +-29 lines by the clip; every body falls inward at speed about c, so that square edge and frame retreat inward as a square front (edge at about 25 / 21 / 17 at ticks 5 / 10 / 14). Later, the outward shell (radius about 52 at tick 71 at 1e5, 35-75 at ticks 57-100 at 1e9) is simply cropped by a square plot window (+-40 for the 100k grids, +-32 for the 1e9 video), which produces the rounded square; with a +-100 window it is a circle in both truth and model. (2) Separately, the model really does have a spurious boundary force: within about 2.5 units of the grid edge (|x| or |y| from about 29.5 to 32) it produces alternating accelerations of order 1e3 to 7e3 (exact gravity there is about 120), and exactly zero beyond 32. This is independent of the particle distribution, tied to the grid edge (it moves with the grid), and comes from both the FFT-kernel gradient and the U-Net. It makes the outward-crossing flow a one-way valve (reversal fraction near the edge 3-4.5% vs 0.15% in truth; inward-moving fraction beyond the edge 0.5% vs 8%; 32 of 400 test tracers thrown to r>70 vs 0 in truth) but it affects a small fraction of bodies and is not the main source of the box seen in the wide panels.

## Setup facts verified in code

- IC (scripts/scatter_bh_large.py:50, scatter_bh.py:50, scatter_bh_1b.py make_ic): `clip(normal(0, sigma), -29, 29)` per axis. No clamp, reflection or wrap anywhere in the roll loops (grep of clamp/clip/reflect/wrap in the four scripts: only the IC clip and the 0.99c velocity clamp in the momentum conversion). Not verified for scatter_bh_timing.py (not used for these runs).
- Speeds: initial per-component sd 0.3*sqrt(N/(4*8)) is 16.8 at 1e5 and 1677 at 1e9, both far above c=10, so velocity is capped at 0.99c at t=0. Measured mean speed from position differences: model 1e5 9.8/10.0/10.0 at ticks 1/10/50; truth 9.65/9.39/9.45; 1e9 subsample 10.00.
- Grid: `scatter_field.py` `_corner_iter`/`_corners4` zero the weight of corners outside the grid (tokens outside get no deposit and gather zero), so force is exactly 0 for |coord| > 32; with momfix a small common offset (2.1 in the probe) remains.
- `neg_grad` pads the potential with zeros (`F.pad`) before the central difference. The U-Net uses zero padding (3x3 convs). `kernel_acc` uses a 2G FFT (no wrap) and then crops to G.
- The 1e9 wide panel is `box_half = 32` (results/scatter_bh_1b_hist.npz), i.e. exactly the grid. The 100k grids (scripts/render_bh_grid.py, `window +-40`) use +-40.
- 1e9 job 3434 used fp16 body state with `--mom-scale 16384` (results/scatter_bh_1b.json). Mesh only (`ops.no_pp`).

## Findings

### A. 100k, truth vs model: the rounded square is a window crop (established)

results/square_wall/square_wall_100k.png, square_wall_100k.json (scripts/square_wall_100k.py).

- Only 75 of 100000 bodies are on the clip lines (34 on |x|=29, 41 on |y|=29), identified by regenerating the IC with seed 4738 (IC regenerated matches the model t0 to 1e-6 and truth t0 exactly). At 100k the clip frame is invisible.
- At tick 71 the shell is at radius about 52 (99th percentile of r: model 52.3, truth 51.5). The +-40 window cuts the shell into the rounded square seen in both the truth and model grids; the +-100 window shows a circle in both (top row, cols 3-4 of the figure). The radial count histograms of truth and model agree on the shell position (r99 at ticks 57/71/100: model 38.6/52.3/81.1, truth 38.0/51.5/79.9).
- Fraction of bodies inside the 32 box at tick 100: model 0.843, truth 0.822.

### B. 1e9 subsample: the t14 box and curtains are the IC support and the clip frame (established for the model; truth at 1e9 untested)

results/square_wall/square_wall_1b.png, square_wall_1b_curtain.json, square_wall_noclip*.png/json, square_wall_noclip_profiles.png.

- Of the 1e6 stored subsample bodies, 601 sit exactly on the clip lines at t0. At t14 all 601 are at r>15 and make up 17% of all bodies at r>15 (601 of 3590); they are 46% of the bodies in the vertical curtain strip (|x| in [16,21], |y|<12; 262 of 567) and 13% in the horizontal one. The red points in square_wall_1b.png (bottom left) trace the curtains.
- Control (job 3447, same model/seed, IC positions beyond +-29 redrawn instead of clamped, `--ic-resample` added to scripts/scatter_bh_1b.py, 30 ticks, fp16 with `--mom-scale 16384`): clip-line bodies 601 -> 3 per 1e6. Curtain-strip counts at t14 fall 567 -> 287 and r>15 counts 3590 -> 2767 (subsample). In the full-N histograms (square_wall_noclip_hist.png) the thin bright frame at t0 and the spikes in the x/y marginals (square_wall_noclip_profiles.png) disappear, but a square-edged cloud and the inward-retreating box remain, because the Gaussian is still truncated at +-29. The marginals show the sharp edge moving 29 -> about 25 (t5) -> about 21 (t10) -> about 17 (t14) (read off the plot, approximate), roughly the inward speed c of the whole cloud.
- So the clip frame is a bright thin addition; the box itself is the square support of the IC. Not tested: a run with a rounder IC (e.g. radial cut) which would show whether the box is entirely the support; and a truth run at N=1e9 (impossible), so that truth shows the same frame is inferred from identical IC, not measured. The 100k truth positions of the 75 clipped bodies trace the same lines at t5 (square_wall_clip_100k.png) but are too sparse to show in density.

### C. Spurious force at the grid edge (established, probe jobs 3444 and 3445)

Scripts scripts/square_wall_probe.py, square_wall_probe2.py (+ analyse scripts); figures square_wall_probe_static.png, square_wall_probe2.png; data square_wall_probe_static.json, square_wall_probe2_summary.json.

- Massless tracers on the +x axis from x=16 to 46 in static N=1e5 scenes (clipped Gaussian, truncated Gaussian, uniform disc r<20), model acceleration (full, and mesh-only) vs exact fp64. Interior (x<=28) model/exact within a few percent to 20%. At x=29-32: +68, -1316, +3053, -960 at x=29,30,31,32 (clipgauss) vs exact -141 to -111; the 0.5-resolution profile has a peak of about +4960 at 30.5 and about -3971 at 31.5. Identical for the three scenes to within a few percent, so it is not a response to the mass distribution or the IC clip. The +y axis shows the same (peak about +6950). For x>=33 the model gives about +2 (momfix offset) vs exact -103 to -50.
- Probe 2 (N=2e4, mesh only, decomposed): edge pattern present in the kernel gradient alone (max 2074) and in the U-Net field alone (max 2460) within [edge-8, edge+2], exact max there 42. With grid 256, extent 128 (weights unchanged, same h) the same pattern appears at x=62-64 (peak +2768); with the scene and tracers shifted by +10 it appears at x=20.5-22 (peak +2023, edge at 22). So it is anchored to the grid edge, not to absolute radius or the mass distribution. Which exact padding mechanism creates it (zero-padded `neg_grad` vs U-Net conv padding) is not separated beyond "both kernel and net parts show it" (untested: padding replacement experiment).
- Consequence on trajectories: 100k model, velocity along an axis (position differences): reversal of outgoing bodies (|v|>1 before and after) at |coord| 28-31: model 4.5%, truth 0.15%; at 31-33: 1.8% vs 0.13%. Mean outward axis velocity jumps 4.2 -> 6.1 -> 7.6 across 28-32 for the model vs smooth 5.9-6.5 for truth; fraction of samples moving inward drops 20% -> 0.7% (model) vs 10% -> 8% (truth) across the edge. 1e9 subsample shows the same pattern (inward fraction 17% -> 0.5%, reversal 3.1% at 28-31).
- Tracer rollout (N=2e4 cluster, 400 massless tracers launched outward at 0.95c from r about 24): all of truth turns back by r about 35 or passes later; model tracers that cross the edge are pushed out: max r of tracers model median 40.8, 99th pct 120, max 125 vs truth median 48.8, 99th pct 60.6, max 61.6; 32/400 tracers beyond r=70 in the model, 0 in truth (square_wall_probe_rollout.png).
- In the 100k run the model has 541/619/799/914 bodies beyond the truth's 99.9th-percentile radius at ticks 57/71/86/100 (truth: 100 by construction), about 0.5-0.9% of bodies thrown ahead of the shell.

### D. fp16 / chunking (1e9 script)

- The 100k run (fp32, no chunked-scatter) shows the same window-crop square, so fp16 and chunking are not needed to explain the visuals. fp16 position spacing is 2^-6=0.016 at |x| in [16,32) (from the format, not measured) against per-tick displacement about 1, so quantisation is not a plausible cause; fp16 vs fp32 at 1e8 agreed within 0.1% on core r45 (earlier commit 79adcc4). Not tested at the edge specifically.
- Unexpected: running scatter_bh_1b.py with the default `--mom-scale 1` in fp16 gave NaN everywhere at tick 1 (job 3446, cancelled-by-completion, results unused) and 8.8 s/tick. Consistent with fp16 momentum overflow (p exceeds 65504 once gravity at N=1e9 acts), not investigated; the scale 16384 is required, as in job 3434.

## Ranked conclusion

Established:
1. The 100k rounded square is the +-40 window cropping a circular shell; truth has it too. Same mechanism for the 1e9 +-32 panel in later ticks (shell radius at ticks 57/71/100: 99th pct 40.5 / 54 / 82.7 from the subsample, larger than the 32 half-width; 6.5% / 13% / 14% of bodies outside the box).
2. The early box/curtains at 1e9 are the truncated-Gaussian IC (square support) plus the clip frame (601 ppm of bodies) retreating inward at about c.
3. The model has a grid-edge force artifact (+-1e3 to 7e3 vs about 120 exact in the last 2.5 units, 0 beyond) independent of the scene; it reflects about 3-4% of outgoing bodies near the edge, and acts as a one-way valve, throwing a small fraction of bodies far ahead of the shell.

Probable:
- The "bounce off walls" impression in the 1e9 video is mostly 1 and 2; the edge artifact (3) is a minor contributor visually. The 1e9 truth would show the same box and curtains (identical IC), not measured.
- At t45-65 in the 1e9 run the shell arrives at the grid edge and 29-32 holds up to 3.6e4 of 1e6 bodies at once (max-norm coordinate bins), then drains; consistent with the edge region seeing the shell (transit time about 3 ticks across the band), not tested as a trap.

Untested:
- A truth run showing the curtains at large N; a radially cut IC; the specific padding cause of the edge force and a fix (pad with extrapolated or zero-gradient potential, wider extent with the same h; the g256 run shows widening the grid only moves the artifact); whether the edge artifact changes the 1e9 core/energy statistics.

## Files

Scripts: scripts/square_wall_*.py, scripts/polaris_square_wall_{probe,probe2,noclip}.sh. Jobs: 3444 (static probe plus tracer rollout, a few minutes), 3445 (decomposition), 3446 (failed, mom-scale 1), 3447 (no-clip 1e9 control, 30 ticks, 80 s loop). `--ic-resample` was added to scripts/scatter_bh_1b.py (it was picked up by commit 8636592). Figures and small JSONs in results/square_wall/. Large npz (noclip snaps/hist) kept out of the repo.
