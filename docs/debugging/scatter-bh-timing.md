# Scatter field on the black-hole scene: wall-clock and memory at N = 1e6..1e8 (timing only)

Script `scripts/scatter_bh_timing.py` (model rollout only, no reference, no accuracy). Scene: Gaussian cluster sigma 8, vfac 0.3, c 10, relativistic kick-drift-kick, dt 0.1, seed 4738 (torch GPU generator), fp32 state, Exp E `ms_kp_pot_v_g128` (grid 128, extent 64, h 0.5), 20 ticks. Hardware: H200 NVL (143.8 GB), one GPU. Per-tick time is CUDA-synchronised and covers kick, drift, force, kick (one force eval per tick); tick 1 is the first tick after one untimed-in-table initial force. Jobs 3428 (1e7), 3429 (1e8), 3430 (1e6); results `results/scatter_bh_timing_<N>_<pair>.json`. Code ran from `~/augur-bhtime` (current master scatter_field.py uploaded; the `cell` opt applied via `sf.apply_opt(model, "cell")`).

| N | pair | ticks | median ms/tick | tick-1 ms | peak GB | frac outside grid (tick 20) | ns/body/tick | status |
|---|---|---|---|---|---|---|---|---|
| 1e6 | none (mesh only) | 20 | 2.9 | 3.5 | 0.18 | 3.8e-4 | 2.9 | measured |
| 1e6 | cell | 11 of 20 | 89,096 (89 s) | 9,355 | 1.69 | 1.7e-4 (tick 11) | 89,096 | stopped at 11 ticks (projected 3612 s for 20); tick time 9 s -> 181 s, still rising |
| 1e7 | none | 20 | 17.7 | 12.0 | 1.43 | 1.9e-4 | 1.8 | measured |
| 1e7 | cell | 0 | - | - | - | - | - | not run: max cell occupancy at t0 97,539, N*9*M = 8.8e12 candidates, estimated 1492 s/tick at t0 alone (extrapolated, over the 900 s skip threshold) |
| 1e8 | none | 20 | 163.4 | 104.6 | 14.04 | 7.4e-5 | 1.6 | measured; no OOM (14 GB of 141) |
| 1e8 | cell | 0 | - | - | - | - | - | not run: max occupancy at t0 975,099, N*9*M = 8.8e14, estimated 1.5e5 s/tick (extrapolated) |

Mesh-only cost is linear in N: 1.6-2.9 ns/body/tick from 1e6 to 1e8 (below 1e6 it is launch-latency floor ~2.5 ms). Memory is about 140 bytes/body (14.0 GB at 1e8).

Cell pair term does not scale on this scene. The cold collapse concentrates bodies: max cell occupancy M (2x2-unit cells) is 9.9e3 at t0 for 1e6 and grows to 2.3e5 by tick 11 (23% of all bodies in one cell). The candidate width is 9M per query, so cost is about N*9*M, i.e. O(N^2) in the collapsed state. The skip estimate uses 1.7e-10 s per candidate, a guess from the 1e5 cold figure (0.45 s/tick, `scatter-scaling.md`); at 1e6 t0 it gave 15 s against 9.4 s measured at tick 1, so it overestimates by ~1.6x at t0 for 1e6 but is a lower bound once M grows. The skipped rows are estimates, not measurements.

1B extrapolation (not measured): mesh-only at the measured 1.6-1.8 ns/body/tick gives about 1.6-1.8 s/tick, and 140 bytes/body gives about 140 GB, which does not fit 141 GB with this implementation (index_add/gather temporaries), so 1e9 would need fp16 state or chunked scatter. Cell pair term at 1e9 is unreachable (N*9*M ~ 9e16 candidates at t0).

Anomalies and caveats:
- vmax/c is 1.0000 (up to 1.0000002, fp32 rounding of p/sqrt(1+p^2/c^2)) from tick 1 at every N: initial speeds (sigma_v = 0.3*sqrt(N/32) = 53 at 1e6) are far above c = 10 and saturate, as at 1e5 in `scatter_bh_large.py`. R_s = 2N/c^2 = 2e4 at 1e6, far outside the box, so this is not a physically meaningful regime; timing only.
- Fraction outside the grid stays below 4e-4 at all N (at 1e5 it reached 0.15 by tick 100); over 20 ticks the cloud is not leaving the box.
- Mesh time rises over ticks 1-10 (12 -> 20 ms at 1e7) as the cloud compacts (index_add contention on fewer cells), then eases.
- No OOM at any N, so no fp16 or fewer-temporary retry was needed.
