# Adaptive far field: wall-clock and memory scaling (H5/H2)

Question: does the learned-estimator dual-tree force (`est`, `scripts/adaptive_force.py`) actually get cheaper than the alternatives in wall-clock, and how does time and memory grow with N?

Setup: one NVIDIA H200 NVL (140 GB), 2D softened gravity, float64 forces. `est` uses lam calibrated so rel_l2 is about 0.5-0.8% (lam 1.79). `geo matched` is the geometric dual tree with theta tuned to the same rel_l2. `geo .35` is the default theta 0.35. `bh` is target-based Barnes-Hut. `mesh` is a particle-mesh solve. Median of 3-5 reps. Results: `results/scaling_*.json`; tables from `scripts/scaling_summarize.py`.

## Force evaluation, flyby state (median s / peak GB)

| N | est | geo matched | geo .35 | bh | mesh (best grid) | exact |
|---|---|---|---|---|---|---|
| 10k | 0.036 / 0.40 | 0.028 | 0.024 | 0.025 | 0.098 | 0.006 |
| 30k | 0.058 / 1.17 | 0.047 | 0.034 | 0.071 | 0.041 | 0.051 |
| 100k | 0.131 / 3.50 | 0.103 | 0.060 | 0.163 | 0.055 | 0.562 |
| 300k | 0.336 / 9.08 | 0.291 | 0.153 | 0.543 | 0.070 | 5.16 |
| 1M | 1.048 / 8.19 | 0.937 | 0.465 | not run | 0.065 | not run (about 57 s extrapolated) |

Uniform state: 1M est 0.949 s, geo matched 0.505 s, geo .35 0.358 s. Est rel_l2 0.0052 (uniform) and 0.0081 (flyby) at 1M.

Log-log exponent of time vs N, N >= 1e5: est 0.90 (flyby) / 0.94 (uniform); geo matched 0.96 / 0.85; exact 2.02. Kernel evaluations per particle are flat: est 240-253 flyby (5% variation), 203-234 uniform then 211 at 1M; geo matched 363-457 flyby.

Findings:
- Est beats exact all-pairs from about 30k (level at 30k, 4.3x at 100k, 15x at 300k, about 55x at 1M).
- Est is not faster than the plain geometric tree at matched accuracy: it is 10-90% slower in wall-clock despite fewer kernel evaluations (flyby: 244 vs 393 per particle at 1M). The per-pair MLP head is the cost: at 100k, estimator 39% of est time, direct 29%, m2l 10%, geom 7%, tree 6%.
- Mesh is faster than est at N >= 100k and roughly flat in N, but rel_l2 is 1.8-4.1% vs 0.5-0.8%; mesh error worsens with N at fixed grid.
- Sub-linear fitted exponents (0.9) reflect GPU under-utilisation at small N, not sub-linear work; cost per particle is constant, so the method is O(N) in work.
- Est peak memory: 3.5 GB at 100k, 9 GB at 300k, 8-10 GB at 1M after the chunking fix (below). Uniform 300k was 12.5 GB before the fix.
- Clustered (t10000 frame, 100k): est 0.104 s, cost 148 per particle, geo matched 0.130 s; est is faster than geo matched here.

## Rollout step time (force + audit)

| N | exact | mesh | geo_audit | adaptive | adaptive_opt |
|---|---|---|---|---|---|
| 100k | 0.562 | 0.055 | 0.094 | 0.112 | 0.101 |
| 1M | not run | not run | not run | 0.918 (7.9 GB) | 0.854 (9.2 GB) |

The 1000-target audit is about 1% of a step at 100k (7.6 ms) and about 8% at 1M (77 ms). The optimised fast audit is slower than the original in eager mode (53 ms vs 7.6 ms at 100k in rollout).

## Memory ceiling (120 GB cap, flyby)

| N | opt steady s / peak GB | orig steady s / peak GB |
|---|---|---|
| 2M | 1.47 / 14.5 | 1.59 / 14.4 |
| 8M | 5.75 / 55 | 6.24 / 55 |
| 12M | 8.57 / 75 | 9.30 / 75 |
| 16M | 11.6 / 90 | 12.6 / 90 |
| 24M | OOM (at 113 GB) | OOM (at 113 GB) |

Memory is about 5.6-6 GB per million particles; the ceiling on one 140 GB GPU is between 16M and 24M. Steady-state is about 0.7-0.8 us per particle.

## Optimisations (semantics-preserving)

`scripts/scaling_opt.py`: verified against the original at N=100k, identical tree, identical kernel-evaluation counts, force differences about 1e-13 (round-off). Speedups at 100k: lighter sync 1.08x, fast tree 1.10x, analytic g_grad 1.12x, geo 1.23x. `torch.compile` gave nothing: it could not build on the Polaris nodes (`Python.h` missing for Triton), so all compile paths run as eager fallbacks. Fast exact and fast audit are slower than the originals in eager mode (0.70x, 0.85x). A real speedup of the MLP head (about 39% of time) needs compile working or a hand-written kernel; not tested.

## Bugs found during the test

- `make_accept` (`scripts/dual_estimator.py`) ran the MLP head over the whole traversal frontier in one call and ran out of memory at N=1M (68.7 GiB allocation). Now chunked at 2M pairs; results unchanged. Every earlier user of the original path inherits the fix.
- `torch.compile` failures crashed the optimised benchmark and rollout; now caught and fall back to eager.

## Predictions vs outcome

- Cost per particle roughly constant: confirmed.
- Est time exponent 1.0-1.15: not confirmed, measured 0.90-0.94 (GPU under-utilised at small N; work per particle flat).
- Est slower than exact until 300k-1M: wrong, crossover is about 30k.
- Audit a large share of step at N >= 100k: wrong for the original audit (1% at 100k, 8% at 1M).

## Not done

- Exact and Barnes-Hut were not run at 1M (extrapolated only).
- No speedup from compile could be measured.
- Single GPU, single seed of the state (flyby / uniform / one t10000 frame); accuracy is rel_l2 against exact on a sampled subset for 100k and above.
