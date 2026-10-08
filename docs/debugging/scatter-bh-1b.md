# Scatter-field black-hole scene at N = 1e9 (mesh-only, 16-bit body state, one H200 NVL)

Script `scripts/scatter_bh_1b.py` (reuses the scene and model of `scatter_bh_timing.py`, see `scatter-bh-timing.md`). Model: Exp E `ms_kp_pot_v_g128` (grid 128, extent 64, h 0.5), weights unchanged, pair term removed (`ops.no_pp`). Scene: Gaussian cluster sigma 8 clipped +-29, vfac 0.3, c 10, dt 0.1, seed 4738, relativistic kick-drift-kick on momentum, ICs generated on the GPU chunk by chunk (the RNG stream differs from the timing script because of chunking, same distribution). Jobs: 3432 (precision check v1 + pilots, fp16 momentum overflowed), 3433 (precision check with momentum scale + 1e9 pilots), 3434 (full run). Hardware: H200 NVL 143.8 GB.

## Design

- State: pos (N,2) and momentum (N,2) in 16-bit; scatter, gather, kick, drift are computed in fp32 per chunk of 5e7 bodies, so temporaries are O(chunk) (about 8 GB at the peak) instead of O(N) (the ~140 B/body of the timing script, which is what made 1e9 not fit). 1e9 bodies of fp32 state would also have fit (16 GB state, 21.6 GB peak measured in the 3-tick pilot); fp16 is used because it was requested and it works.
- Half-kicked state: the stored momentum is M = p(t) + dt/2 a(t). A tick is drift with v(M), scatter, grid force, M += dt (a - mean a). This is the same map as kick-drift-kick with the adjacent half-kicks merged. The mass-weighted mean a (momfix) is computed from the grid (gather is the transpose of scatter: sum_i a_i = sum_cells acc W, W the scattered mass), so no extra gather pass. Diagnostics use v(M), half a kick ahead of p.
- Check of the chunked force against `model.force` on 2e5 bodies: max|a_chunked - a_model| / max|a| = 1.2e-4 to 2.1e-4 (fp32 summation order; comparable to the 1e-3 graph/eager mismatches in earlier docs).
- Momentum range: in this scene the IC momenta are ~1e4 (vel is not clamped, only the gamma factor is, see caveats) and grow to |p| ~ 1e8 at N=1e9 (a scales with N). Plain fp16 momentum overflows (65504) on tick 1 at N=1e8 (job 3432: 87% of bodies inf/NaN at tick 1). Momentum is therefore stored as p / S in fp16 with S = 16384 for the 1e9 run (max |p| seen 1.06e8 = 6.5e3 stored; no non-finite values on any tick). bf16 needs no scale.
- Positions in fp16: spacing 0.0156 for |x| in [16,32) and 0.031 in [32,64), vs cell 0.5 and displacement up to c dt = 1 per tick. Bodies leaving the box (14% by tick 100) keep fp16 range (65504), no overflow.

## Precision check, N = 1e8, 20 ticks (job 3433, `results/scatter_bh_1b_chk1e8_s256_check.json`, momentum scale 256)

Reference = fp32 pos + fp32 momentum, same ICs, same code path. tv128/tv1024 = total-variation distance between the 128x128 / 1024x1024 full-box density histograms of all 1e8 bodies (fraction of bodies in different bins).

| state (pos:mom) | tv128 t=1/10/20 | tv1024 t=20 | core r45 t=20 (ref 1.5749) | r90 t=20 (ref 5.603) | mean KE t=20 (ref 1.9291e7) |
|---|---|---|---|---|---|
| fp32:fp16 | 6e-5 / 1.2e-4 / 3.4e-4 | 0.0038 | 1.5750 | 5.602 | 1.9293e7 |
| fp32:bf16 | 1.7e-4 / 3.1e-4 / 1.3e-3 | 0.0132 | 1.5740 | 5.604 | 1.9291e7 |
| fp16:fp16 | 7.8e-4 / 5.5e-4 / 8.1e-4 | 0.0068 | 1.5748 | 5.602 | 1.9291e7 |
| fp16:bf16 | 7.9e-4 / 6.0e-4 / 1.65e-3 | 0.0142 | 1.5741 | 5.603 | 1.9291e7 |

vmax/c is 1.0000 and frac outside the grid is identical to 3 digits in all variants. pmax differs by < 0.7% (bf16) and < 0.05% (fp16 with scale). fp16 positions put about 0.07% of bodies in a different 0.5 cell-scale bin from tick 1 on (quantisation noise, does not grow); fp16 momentum adds a little more by tick 20; bf16 momentum is worse than scaled fp16 (8 vs 11 mantissa bits). What fp16 changes: per-body positions at the level of 0.016 to 0.03 and momentum direction at the level of 1e-3; no change visible in core radius (<0.1%), r90, mean KE, vmax/c, or the coarse density. Chaotic divergence of individual trajectories over 100 ticks was not measured (no 100-tick fp32 run at 1e8), and there is no exact reference at all.

## Results (job 3434, `results/scatter_bh_1b.json`)

| N | state dtype | ms/tick median | first tick | ticks | total wall (loop incl. diagnostics) | peak GB | frac outside grid (t=100) |
|---|---|---|---|---|---|---|---|
| 1e8 (timing script, job 3429) | fp32 | 163.4 | | 20 | | 14.0 | |
| 1e9 pilot (3433, 3 ticks) | fp32 pos+mom, chunked | 1035 (1003 / 1035 / 1161) | 1003 | 3 | 3.6 s | 21.6 | 0 |
| 1e9 pilot (3433, 3 ticks) | fp16 pos+mom/16384, chunked | 1067 (1035 / 1067 / 1193) | 1035 | 3 | 3.7 s | 14.0 | 0 |
| 1e9 full | fp16 pos+mom/16384, chunked | 1325 (min 1035, max 1793) | 1035 | 100 | 263 s (sum of tick times 137 s; the rest is per-tick histograms, subsample and diagnostics; plus 1.0 s initial force) | 16.1 | 0.144 |

Per-tick time rises from 1.03 s to 1.3-1.8 s over the first ticks while the core collapses (dense-bin atomics in the index_add scatter), then settles near 1.3 s. 1.33 ns/body/tick median, against 1.6 ns at 1e8 fp32 in the timing script (the chunked 5e7 scatter/gather is cache-friendlier than the 1e8 monolithic one). The naive extrapolation was ~1.7 s/tick and ~140 GB (does not fit); measured 1.33 s/tick and 16.1 GB. The 1e9 run has no checkpoints by design: total compute is ~4.5 minutes and the state is 1e9 bodies; a crash costs a rerun.

Dynamics: core radius (45th percentile of the distance to the median centre, on the 1e6 subsample) 8.75 at t=0, 1.18 at tick 11, 1.96 at tick 100. 14.4% of bodies are outside the +-32 grid at tick 100 (first exits near tick 50) and feel zero force there (stock model behaviour).

## Outputs

- `results/scatter_bh_1b.json` per-tick ms, frac outside, vmax/c, pmax, mean KE, core r45/r90, peak GB. Also pilots `results/scatter_bh_1b_pilot1e9_{fp32,fp16}.json`, precision check JSON.
- `results/scatter_bh_1b_hist.npz` (not committed, 402 MB): `full` (101,1024,1024) int32 over +-32, `zoom4`, `zoom1` the same over centre +-4 and +-1 (centre = median of the subsample per tick, `centre`), all 1e9 bodies, y rows. `results/scatter_bh_1b_snaps.npz` (not committed, 816 MB): `pos` (101,1000000,2) float32, `steps`, `sub_idx`; fixed uniform random 1e6 subsample.
- `videos/scatter_bh_1b_lensed.mp4` (render_nbody_lens.py, no blur, 1080x1080, 101 frames) and `videos/scatter_bh_1b_density.mp4` (`scripts/render_bh_1b_density.py`: full box, +-4, +-1 panels, log colour per panel, 101 frames). Both decode clean (`ffmpeg -v error -f null -`).

## Caveats

- Mesh-only: no pair term, no exact reference; nothing here is an accuracy result. Only timing, memory, stability and fp16-vs-fp32 self-consistency.
- Unphysical IC regime (already documented): the IC velocity std is 0.3 sqrt(N/(4 sigma)) = 1677 at N=1e9, far above c=10; the speed is clamped for the gamma factor but the momentum is not, so |p| ~ 1e4-1e5 and v = c p/|p| saturates at the cap from tick 1 (vmax/c = 1.0000 on every tick). The scene was not changed.
- Lensed video uses a 1e6 uniform subsample of 1e9 bodies: lensing density is the same up to normalisation, with shot noise ~1/sqrt(local count) (fewer than 1e3 times fewer points); the density video uses all 1e9.
- Diagnostics' velocity is v(M), half a kick ahead of p(t).
- The coarse-grain claims are from a 20-tick 1e8 check; the 1e9 fp16 trajectory itself was not compared against fp32 beyond the 3-tick pilot (core r45 5.7566 vs 5.7565, mean KE 3.73784e7 vs 3.73792e7 at tick 3).
