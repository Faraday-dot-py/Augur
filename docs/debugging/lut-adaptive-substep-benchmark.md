# LUT force tables and adaptive substeps: benchmark (2026-09-25)

Uncommitted. Conservative-contact model (`results/cons_pure.pt` = remote `checkpoints/cons_pure.pt`, 8 Verlet substeps, cell graph)
and, secondarily, CentralForceDynamics near field (cutoff 4). Polaris H200, job 2978 (4 min 14 s GPU wall, log `~/bounce/bounce-lut-bench-2978.log`), seed 4738.
No tracked file changed: variants are installed as an instance-level `forward` (`scripts/lut_adaptive.py`); scripts `lut_bench.py`, `lut_eval.py`,
`lut_central.py`, `lut_summarize.py`, `lut_followup.py` (written, NOT run, see Open), `polaris_lut_bench.sh`; raw results `results/lut_{bench,eval,central}.json`.

Variants: `lut<N>` = N-interval table of `pen*MLP(pen)*100` (pair pen in [0,1], wall pen in [0,2], linear extrapolation above), indexed by the
symmetric pen so pair antisymmetry is exact; suffix `c` = cubic (Catmull-Rom). `skin<M>`: adaptive substeps, ball active if any neighbour has
d <= 2r + M*(|vi|+|vj|)*dt or a wall is within r + M*|v|*dt + 0.5*g*dt^2 (skin0: d<=2r, wall<=r); inactive balls take one exact ballistic+gravity step,
active balls take the 8 substeps among active balls only.

## Regimes
Sparse: uniform, density 0.01 (grid 10*sqrt(N)), speeds +-2.3, timed after 5 steps (gravity on; vmax ~23-30, contact fraction ~2%).
Dense: floor pile, square lattice spacing 1.45 (<=30 rows, grid ~ 1.45*N/30), at rest, timed after 20 steps (contact fraction 0.96-0.985, ~9 neighbours within 2r).
Timing = mean of 10 steps (3 at 1M), synced. Cell graph used at all N.

## Profile (baseline, synced per-section ms per step, sums exceed ms/step because of sync overhead)
| regime | N | ms/step | wall (feat+MLP) | graph | pair MLP | pair geom+scatter | update | kernels/step, GPU-kernel ms |
|---|---|---|---|---|---|---|---|---|
| sparse | 1k | 9.2 | 1.4 | 5.0 | 0.8 | 0.9 | 0.3 | 1163, 2.4 |
| sparse | 10k | 10.7 | 1.7 | 7.0 | 0.9 | 1.1 | 0.4 | 1334, 4.0 |
| sparse | 100k | 14.4 | 4.8 | 8.0 | 1.0 | 1.1 | 0.4 | |
| sparse | 1M | 73 | 38.7 | 32.8 | 1.0 | 1.2 | 0.5 | |
| dense | 1k | 9.5 | 1.6 | 6.0 | 0.9 | 1.1 | 0.4 | 1154, 2.6 |
| dense | 10k | 10.7 | 1.6 | 7.0 | 1.2 | 1.1 | 0.4 | 1307, 5.0 |
| dense | 100k | 18.9 | 4.8 | 10.8 | 8.2 | 1.2 | 0.4 | |
| dense | 1M | 150 | 38.4 | 61.9 | 77.3 | 6.8 | 0.5 | |

N <= 10k is launch/Python bound: ~1.2-1.3k kernel launches per step, GPU busy 2.4-5 ms of ~9-11 ms. Nothing that cuts arithmetic helps there.
At 1M the graph rebuild (9 per step) and the MLP evals dominate: wall MLP runs on all N*4 values every substep even though pen is 0 for nearly all
(sparse: 38.7 of 73 ms); pair MLP dominates dense.

## Speedup, conservative model (ms/step, speedup in brackets)
| regime | N | baseline | lut1024 | skin1 | lut1024+skin1 |
|---|---|---|---|---|---|
| sparse | 1k | 9.22 | 9.26 (1.00) | 10.29 (0.90) | 10.25 (0.90) |
| sparse | 10k | 10.72 | 10.48 (1.02) | 11.34 (0.95) | 11.39 (0.94) |
| sparse | 100k | 14.43 | 11.82 (1.22) | 14.25 (1.01) | 12.59 (1.15) |
| sparse | 1M | 73.2 | 37.8 (1.93) | 46.1 (1.59) | 28.6 (2.55) |
| dense | 1k | 9.46 | 9.44 (1.00) | 10.50 (0.90) | 10.44 (0.91) |
| dense | 10k | 10.72 | 10.51 (1.02) | 12.30 (0.87) | 12.15 (0.88) |
| dense | 100k | 18.9 | 13.2 (1.43) | 27.0 (0.70) | 21.3 (0.89) |
| dense | 1M | 149.8 | 65.0 (2.30) | 260 (0.58) | 175 (0.85) |

Other variants (same JSON): lut256/lut4096 = lut1024 timing (table size irrelevant); cubic (`lut1024c`, `lut4096c`) slower than linear at every N (0.77-0.81x at
<=10k, 1.19-2.05x at 100k-1M) because of extra gathers/launches; skin0 5.0x at sparse 1M but wrong (below); skin2 slower than skin1 (active fraction 0.86-1.0).
Adaptive active fractions (skin1): sparse 0.49-0.68, dense 0.99-1.0; the falling gas has speeds ~10-30 so the per-pair margin (|vi|+|vj|)*dt is 3-9 units.
Adaptive costs an extra graph build at radius 2r + 2*vmax*dt plus a `vmax` host sync per step; when everything is active it is pure overhead (dense).

## Accuracy vs the unmodified model (n=20/100 eval seeds 9000/12000, ball_1k-style 1000 balls n=317 seed 4738, pile 1000 balls n=100)
Table quality: lut256 max abs err 3.6e-3 (4e-6 of peak force ~800), lut1024 3e-4, lut4096 3e-4 (fp32 floor). Cubic showed a LARGER max error (lut1024c 0.079, lut4096c 0.02)
which scales like O(h), consistent with a boundary-segment artefact of my one-sided end tangents (untested; interior error follow-up not run).

| metric | base | lut1024 | lut4096 | lut4096c | skin0 | skin1 | skin2 | lut1024+skin1 |
|---|---|---|---|---|---|---|---|---|
| err@5/10/20, 4 balls | .1231/.3145/2.3896 | same/same/2.3895 | same | same | .129/.900/7.25 | same/same/2.391 | same/same/2.391 | same/same/2.391 |
| err@5/10/20, 100 balls | .1131/.2980/1.9033 | 1.9039 | 1.9031 | 1.9033 | .145/.479/6.32 | 1.9031 | 1.9028 | 1.9017 |
| E/ball @100 (100 balls) | -390.6 | -393.6 | -407.2 | -392.8 | +3.1e9 | -406.5 | -411.0 | -401.7 |
| y-wall restitution s=3..80 | 1.017 .. 1.005 (1.04 peak) | identical to 3 d.p. except s=40/60: 1.009/1.005 (base 1.011/1.012) | same | same | 1.1, 2.1, 1.05, 2.5, 2.0, 1.55, 3.2, 1.19, -7.3 | = base | = base | as lut1024 |
| x-floor dE s=40/60/80 | 8.8 / 38.8 / 26.2 | 7.6 / 13.4 / 31.4 | 7.6 / 13.4 / 31.5 | 7.6 / 13.4 / 31.5 | 9056 / 850 / 7766 | = base | = base | as lut1024 |
| head-on relKE s=3..80 | 1.00 ... 0.66 (s=30) ... 1.05 | identical (<= 0.001) | same | same | 6.03, 2.32, 1.21, 1.13, 1.0, 1.06, 1.0, 1.0, 1.0 | = base | = base | = base |
| pair momentum err | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |

(Truth x-floor dE = 3.2/0.1/-9.0/21.5 at s=20/40/60/80 from the earlier doc; LUT differs from base at s=60/80 only, hypothesised cause below.)

One-step max |dv| vs base (velocity units; median ball |dv| = 1.34 in sparse, 3.5 in dense):
| state | contact frac | lut256 | lut1024 | lut4096 | lut1024c | skin0 | skin1 | skin2 |
|---|---|---|---|---|---|---|---|---|
| sparse t0 | 0.07 | 3.9e-4 | 2.9e-4 | 1.9e-6 | 1.7e-5 | 2.95 | **1.23** | **1.23** |
| mixed t100 | 0.05 | **2.5** | **2.5** | **2.5** | **2.5** | 101 | 2.3e-5 | 6.9e-6 |
| dense t20 | 0.96 | 6.6e-4 | 3.4e-4 | 1.4e-4 | 1.2e-4 | 6.5 | 1.2e-4 | 1.2e-4 |

Two unexplained single-ball outliers (one ball of 1000): all LUTs at mixed t100 give max |dv| 2.5 independent of table resolution/cubic (mean 7e-3), while skin1 alone is 2e-5.
Hypothesis (UNTESTED): a ball deep past a wall (pen > 2 = wall-table range) gets linearly extrapolated table force instead of the MLP; same cause probably behind the s=60/80
x-floor dE differences. Fix = wall table range >= 8. Second outlier: skin1/skin2 at sparse t0 (max |dv| 1.23 ~ one gravity step, mean 2.5e-3), cause unknown; both `lut_followup.py` diagnostics would answer these.

500-step rollouts, 1000 balls (single seed 4738, chaotic; E/ball = KE - g*x):
- sparse_1k: truth E@100/500 = -1141/610. base -1181/514; lut256/1024/4096/1024c/4096c: -1179/427, -1210/395, -1202/444, -1203/427, -1203/555; skin1/skin2: -1208/499, -1202/408; lut1024+skin1 -1160/537. Spread across variants is +-20% at 500 (base itself is 16% off truth); no systematic drift signature. KE@500 2154-2318 vs base 2276, truth 2344. Contact frac 0.06-0.09 vs truth 0.08.
- pile_1k: truth -812.6@100, -810.1@500; all variants -812.7..-815.1 / -809.7..-811.5 (base -814.3/-810.9). Drift 100->500 <= 4.4 for every variant. KE 15.6-17.5 vs truth 18.0/16.7.
- skin0 blows up in both (E 6e9 at step 100 sparse; 3.5e4 pile), finite but unphysical (balls tunnel and get huge late-contact forces).
- Position divergence from base (mean |dp|): sparse: 8e-5 (lut4096) .. 1.7e-2 (lut256) at step 20; 130-138 by step 100 for every variant (= err vs truth 130-138, i.e. fully chaotic); pile: 1.0-2.0 at step 20 (0.45 skin1), ~9 at 100, ~20 at 500 (base err vs truth 9.4 at 100). So divergence is at the chaos floor for any perturbation (even lut4096c); I did not run a base-vs-1e-6-perturbed-base control (untested), but the LUT variants are indistinguishable from each other in divergence and from base in statistics.
- Conservation: symplectic exactness is not the issue in practice; mixed substep counts (skin1/2) show energy statistics identical to LUT-only. A LUT force is a piecewise-linear central force of the symmetric distance, so momentum is exact by construction; no potential is exactly consistent with it (small, not separately measured).

## CentralForceDynamics near field (cutoff 4, dt 0.1, MLP force in log d; table on log d in [ln 1e-3, ln 8])
Step time (2 accels), ms, strips=1 (4 at 1M):
| regime | N | neighbours/body | base | lut1024 | lut4096c |
|---|---|---|---|---|---|
| sparse | 1k | 3.9 | 2.1 | 2.0 (1.05) | 2.2 (0.94) |
| sparse | 10k | 4.0 | 2.4 | 2.4 (0.98) | 2.5 (0.94) |
| sparse | 100k | 4.0 | 3.8 | 3.0 (1.26) | 3.3 (1.15) |
| sparse | 1M | 4.0 | 25.2 | 17.1 (1.47) | 18.0 (1.40) |
| dense (4x compressed) | 1k | 57 | 2.1 | 2.0 (1.04) | 2.2 (0.93) |
| dense | 10k | 62 | 4.0 | 2.5 (1.56) | 2.8 (1.43) |
| dense | 100k | 64 | 21.2 | 8.8 (2.41) | 10.0 (2.12) |
| dense | 1M | 64 | 203 | 77.9 (2.61) | 88.6 (2.30) |
Profile of one accel (1M dense): graph 25 ms, MLP 67 ms, gather/scatter 5 ms; sparse 1M: 5.4 / 4.3 / 0.4. (The sparse 1k "graph 16.6 ms" row is a CUDA warm-up artefact.)
Adaptive substeps do not apply (no substeps). Accuracy: table error lut1024 1.5e-5 of peak force (lut256 2.3e-4, lut4096 1.2e-6, cubic 6e-7). One-step max |dp| <= 1.6e-4, max |dv| <= 1.6e-3 in every
case (dense 10k is 1.55e-3 for all sizes and cubic, i.e. fp32 summation-order noise, not the table). 200-step divergence from the MLP run is chaotic in the dense/clustered case (mean |dp| ~ 18-26,
max ~140, KE +-1..10%; sparse 10k lut4096: mean 0.07, KE 2e-5 relative) and I did not separate it from a noise-floor control (untested). Momentum stays ~4e-4 in all runs.

## JS proxy (node v24, my own JS port of the conservative step with RANDOM 64x64 MLP weights, cell grid, 8 substeps; NOT web/js/model.js, which is the soup-B attention/GRU model, so LUT/adaptive do not apply to the shipped web model as is)
ms/step, single thread. JS baseline evaluates the wall MLP only where pen > 0 (unlike torch), so it is the stronger baseline.
| regime | N | base | lut1024 | skin1 | lut1024+skin1 |
|---|---|---|---|---|---|
| sparse | 1k | 4.1 | 0.91 (4.5x) | 4.7 (0.87x) | 1.4 (2.9x) |
| sparse | 10k | 21.0 | 10.2 (2.1x) | 25.3 (0.83x) | 9.1 (2.3x) |
| dense | 1k | 118-135 | 2.5-3.1 (~45x) | 118-134 (1.0x) | 5.8-7.4 (~18x) |
| dense | 10k | 2429 | 23.9 (102x) | 2302 (1.06x) | 72 (34x) |
In JS the 4.3k-MAC scalar MLP per pair/wall evaluation dominates, so LUT is decisive (dense: 20-100x). Adaptive gives nothing (active 0.54-1.0). Proxy only: random weights, N<=10k, one run each.

## Recommendation
Decision (user, 2026-09-25): the LUT is NOT the main method. It tabulates the learned force as a function of one scalar (pair/wall penetration, or log d), which bakes in a 1D central-force assumption and discards whatever the MLP would learn beyond it (extra inputs, non-central terms, other force laws). Acceptable only as an optional inference cache for CentralForceDynamics / the conservative-contact model; benchmark kept as a record. Points 1-4 below are the measured options, not the adopted plan.

1. Adopt a linear LUT (1024 entries, shared MLP-free path) for the pair and wall forces: accuracy cost is below noise (one-step <= 3e-4, err@k unchanged to 4 digits, restitution/relKE unchanged, energy/contact statistics unchanged), momentum stays exact. Real GPU gain only at N >= 100k: 1.2x (sparse 100k), 1.9x (sparse 1M), 1.4x (dense 100k), 2.3x (dense 1M); ~1.0x at N <= 10k (launch bound). Large win in JS/CPU. Before adopting, widen the wall table range to >= 8 (pen) and check the two outliers (untested fix); use lut4096 if you want 1e-6 one-step agreement (same speed). Cubic buys nothing.
2. Do not adopt skin0 (unstable: energy 1e9, restitution and head-on relKE wrong). skin1 is statistically indistinguishable in the rollout/eval metrics but is a net loss except sparse N >= 1M with LUT (2.55x vs 1.93x for LUT alone), loses 5-40% in dense and at small N, has one unexplained one-step outlier (1.23), and complicates the tiled path. Not worth it as is.
3. Better untested targets from the profile: rebuild the graph once per step with a skin margin (Verlet list) instead of 8x (graph = 33-62 ms of 73-150 ms at 1M), and skip wall evaluation for balls with no wall within r (wall = 38 ms at 1M even with pen == 0); CUDA graphs / torch.compile for N <= 10k where launch overhead is 60-75% of the step.
4. CentralForceDynamics: LUT is worth it at N >= 100k (1.3-1.5x sparse, 2.4-2.6x dense) with one-step-lossless accuracy; long-run agreement not separated from chaos.

## Open (blocked by the Polaris priority hand-off to the flyby job; not run)
`scripts/lut_followup.py` (upload + `PYTHONPATH=. python scripts/lut_followup.py`, ~1 min): interior table error, wall range hi in {2,8,32} effect on the mixed_t100 outlier and pen at the outlier ball, and the sparse_t0 skin1 outlier's neighbours/wall distances.
