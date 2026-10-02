# Scatter-field architecture (2D gravity)

Uncommitted. Files: `scripts/scatter_field.py` (model, sim helpers, probes), `scripts/train_scatter_field.py` (train/eval/summarize), `scripts/polaris_scatter_field.sh` (job).

## Spec and implementation

Tokens `{x,y,dx,dy,m}`. Each step:
1. CIC (bilinear) scatter of `[m, m*dx, m*dy]` (density, /cell area) onto a G x G grid covering `extent` units (x,y consumed by the scatter).
2. Input to the grid net = scatter channels + (if recurrent) the previous output field. Output = new diff field `[ddx, ddy, hidden...]`. `dm` is hard-zero: m is conserved in the data so there is no dm channel.
3. Bilinear gather of the output field at each token's x,y -> `dv = dt * gathered`; `v += dv`; `x += dt*(v_old + v_new)/2`.
4. Repeat. O(N + G^2) per step, no all-pairs.

Nets: `ss` = 8-layer residual 3x3 conv, width 32 (local, translation-equivariant, per-step receptive field measured by autograd impulse, reported in results as `rf_units`). `ms` = U-Net (avg-pool down, nearest up, skips; 4 levels, fewer at G=32; equivariant only to shifts by 2^levels cells) -> large/global receptive field per step. Zero padding; tokens outside the grid get zero force.

Deviations / choices (all mine, not in the spec):
- Output channels are acceleration-like (scaled by dt); positions relative to CENTER=500 internally.
- Optional extra hidden channels in the diff field (`ss_rec_h8`, 8 extra); default has none.
- Momentum fix (`momfix`, default on): subtract the mass-weighted mean `dv` from all tokens, so `sum m*dv = 0` exactly (this also cancels any net self-force). Ablated off (`ms_rec_nomomfix`). Per-token self-force is NOT removed (it is not separable in a nonlinear net); measured by `self_force_probe` (one isolated token at rest, momfix off, true force 0; reported as mean |a|).
- Loss/curriculum as the baseline script: pos MSE + 0.1 vel MSE, k 4 -> 20 over training; Adam 1e-3 cosine to 5%; 2000 iters, batch 32 (baseline: constant lr, 3000 iters, batch 16, per-scene).
- Baseline = `CentralForceDynamics` retrained on identical data (train seed 4738), also evaluated in my harness so conservation metrics are comparable.

## Experiments

Data/truth/eval reuse `scripts/gravity_sim.py` (`make_dataset`, torch fp64 truth, softened G=1, eps 0.5, dt 0.1, 4 substeps). Held-out eval seeds 9000/12000 (the same ones as earlier gravity results; 4738 is the train seed), 48 scenes, 100 steps (err@5/10/20 are headline; 50/100 are beyond the 30-step train windows). err = mean per-body position error in sim units (see reference_gravity_error_units).

- Exp A: 2 unit-mass bodies, extent 32. Exp B: 10-100 unit-mass bodies, `scale_init`, extent 64. Exp C: star (mass 8-30) + 3-11 light bodies (0.01-0.5), mixed masses, extent 64; baseline cannot be run (takes no masses).
- Ablations: recurrent vs not; local vs U-Net; grid G=32/64/128; momfix off; hidden channels.
- Delete-the-star (Exp C): static star mass 20 at centre, 4 fixed test masses (0.01) at distances 3..28. Tracers/star held in place, 12 warm steps, then the star token is deleted, 12 more steps. `lag` = number of post-deletion steps until the inward acceleration of a tracer drops below 50% of its pre-deletion value (truth: 0, Newtonian gravity is instantaneous). Control run (star kept) checks the trace is stable.

## Predictions (written before any result)

1. Exp A, 2-body: the baseline (exact central force, Verlet) wins on err@k: it has the right inductive bias. Best scatter variant err@20 within 3-10x of baseline; G=128 beats G=32 clearly (sub-cell resolution: eps=0.5 vs cell 1.0 at G=32 over extent 32 -> cell 1.0, G=64 0.5, G=128 0.25). Scatter errors grow faster at later steps.
2. Exp B, 10-100 bodies: baseline still better at err@20 (it is all-pairs exact-form; scatter is a smoothed mesh force) but the gap narrows as N grows because mean-field dominates; scatter variants will beat constant velocity by a wide margin.
3. Recurrent vs not: small effect on err@k for the U-Net (it already sees everything); for the local net, recurrent should help noticeably (information travels beyond the per-step receptive field over steps).
4. Local vs multi-scale: the local net (RF ~9 cells ~ 9-18 units) will be worse than the U-Net where separations exceed its RF; Exp A at extent 32 less so than B at 64.
5. Momentum: with momfix, momentum drift is ~1e-7 (float32 round-off); without, drift is clearly non-zero (self-force + non-antisymmetric learned field). Energy drift: not conserved by construction; I expect it larger than baseline (symplectic-ish central force) and to grow with horizon, esp. past step 30. 2-body orbits: scatter orbits precess/spiral (non-conservative), baseline stays stable.
6. Self-force probe: non-zero for all variants (mean |a| a few % of a typical 2-body accel), worse at coarse grids.
7. Delete-the-star: non-recurrent U-Net: lag 0 at all distances (instant, within a global RF). Non-recurrent local: lag 0 inside the RF, and no force at all (not even a pre-deletion force) outside the RF so it is not informative beyond. Recurrent local: pre-deletion force reaches farther than RF because of accumulated propagation; after deletion the stale field persists -> lag grows roughly like d/RF_units (one step per RF-radius), i.e. the user's d/(c dt) with c = RF/dt. Recurrent U-Net: lag small (0-1) because the net learned from Newtonian training data to rely on the fresh scatter, but may show a 1-step tail from stale state. Caveat: training data contains no deletions and Newtonian truth has lag 0, so any lag is an artifact of the architecture, not something that was trained.

## Results

See "Results (measured)" at the end of this file; predictions above are unedited.

## v2 extension (coordinator decisions 2026-10-01): `MultiLevelScatterField`

Implemented in `scripts/scatter_field.py`; run results below.

- Moments scatter (NGP into the token's cell, per level): `[m, m*o (dipole), m*o o (quad, flag), m*v, n]`, o = offset from cell centre in cell units.
- Field per cell: `[a0 (2), Jh = h*dA/dx (4), dm (1), hidden]`; token accel = a0 + J.(x - x_c) at its leaf cell. All output heads zero-init; dm kept as a channel (mass += dm) and trained to 0 via a mass-error loss term.
- Block-sparse multilevel: levels l=0..L-1, cell h_leaf*2^l, 8x8 tiles allocated only where tokens with leaf <= l exist; top level covers the domain. Moments restricted upward (direct per-level deposit = exact sum), field prolonged downward with Taylor (a0 + J.delta, J halves). Per-level net = 3 tile convs with 1-cell halo exchange from neighbour tiles.
- Adaptive leaf: coarsest level whose cell holds <= k_leaf tokens, capped at level 0 (h_leaf = 0.5). Dense int count maps are used as transient index structures only; all feature/field data is sparse tiles.
- Persistent diff field: previous per-level (tile map, data); new tiles are initialised from the prolongation of the coarser level's current field.
- Diagnostics added: per-channel RMS (pos/vel x,y at 5/10/20), curl-of-force-field probe (static star + light test token lattice; curl/total ratio, truth 0), self-force probe, dm drift, Exp D = 2-body bound orbits d = ratio*h_leaf, ratio log-uniform in [0.25,16], with a fixed-ratio sweep (0.25..16) of err@k, separation error, energy/L drift.
- Variants: ml_rec, ml_norec, ml_noquad, ml_k4 (A, B, C, D subsets); baseline CentralForce also swept.

Predictions for v2 (before running): (1) ml_rec err@20 on Exp A within 2x of ms_rec, better at d < 2 h_leaf thanks to dipole/quad; (2) sweep breaks (sep error > 50% of d at step 100) at d/h_leaf <= ~0.5-1 for ml_rec, ~1-2 for the dense ss/ms nets; baseline never breaks; (3) curl/total ~0.1-0.3 for all learned fields (NGP + Taylor is discontinuous at cell edges; nothing enforces curl-free); (4) mass drift ~1e-3 or smaller; (5) leaf-level switching introduces force jumps visible as energy drift.


## Results (measured)

Jobs: smoke 3194 (failed, device bug), 3196/3199 (smoke ok, v1/v2); Exp A v1 3197; Exp C v1 3198; Exp D + A/C v2 3200; Exp B 3201; reruns 3202, 3208 (resume/eval fixes); ml_nodm (dm not applied) 3209. 4000 iters (B: 3000), batch 32, seed 4738, eval seeds 9000/12000 (48 scenes). Plots in `videos/scatter_field_*.png`; raw summaries `results/scatter_field/{A,B,C,D}_summary.md` (remote `~/bounce/results/scatter_field`, local copies in `~/polaris-mcp-files/sf/res`). v1 = dense single-grid (`ss`=local conv, `ms`=U-Net); `ml` = v2 multilevel.

err@5/10/20 (seed 9000; 12000 agrees within ~20%). Const-vel: A 0.011/0.045/0.168, B 0.071/0.274/0.956, D 0.073/0.241/0.571.

| model | A (2 body) | B (10-100 body) | D (orbits, d=0.25-16 h) |
|---|---|---|---|
| CentralForce baseline | 0.0001/0.0003/0.0007 | 0.0007/0.0021/0.0064 | 0.0008/0.0012/0.0027 |
| ss_rec (local conv, recurrent) | 0.0013/0.0023/0.0098 | 0.017/0.045/0.125 | 0.0024/0.0041/0.0090 |
| ss_norec | 0.0020/0.0072/0.0271 | 0.018/0.060/0.188 | - |
| ms_rec (U-Net) | 0.0010/0.0019/0.0098 | 0.016/0.044/0.122 | 0.0027/0.0044/0.0097 |
| ms_norec | 0.0017/0.0036/0.0101 | 0.017/0.051/0.140 | - |
| ms_rec G=32 / 128 (A: h=1/0.25) | 0.0024/.0072/.0177 ; 0.0005/.0009/.0031 | 0.046/.133/.358 ; 0.0047/.0106/.0330 | - |
| ml_rec (v2) | 0.0055/0.0158/0.0503 | 0.019/0.063/0.204 | 0.014/0.040/0.117 |
| ml_norec | 0.0046/0.0178/0.0591 | 0.019/0.066/0.209 | - |
| ml_noquad | 0.0062/0.0192/0.0600 | - | 0.018/0.044/0.130 |
| ml_k4 | 0.0088/0.0339/0.1337 | 0.046/0.165/0.498 | 0.057/0.115/0.216 |
| ml_nodm (dm not applied) | 0.0057/0.0187/0.0539 | - | 0.015/0.036/0.116 |

Longer horizon, drift (A, step 100, seed 9000): err@100 baseline 0.009; ss_rec 0.32; ms_rec 0.23; ms_rec_g128 0.091; ml_rec 1.11; ml_noquad 2.5; ml_k4 6e9 (blow-up). Energy drift |dE/E|@100: baseline 0.010, ss_rec 0.43, ms_rec 0.29, g128 0.10, ml_rec 1.5, ml_noquad 177, ml_k4 2e20. Momentum |dP|@100 (truth masses): baseline 3e-17; all v1 with momfix ~5e-8 (float32 round-off); ms_rec_nomomfix 6.6e-2; ml_rec 6.6e-2 and ml_noquad 1.9, ml_k4 4.6e9 (cause: dm drifts the mass, momfix uses the drifted mass); ml_nodm 2.5e-8. Angular momentum drift dL/L@100: baseline 0.001, ss_rec 0.31, ms_rec 0.20, g128 0.07, ml_rec 2.1. Mean dm drift @100 (A): ml_rec 0.031, ml_norec 0.006, ml_noquad 0.42, ml_k4 0.39.

### What the numbers say
1. Prediction 1/2 confirmed in direction, wrong in size: the exact-form baseline wins everywhere, by 10-100x at 2-body (err@20 0.0007 vs best dense 0.0031 at G=128, 0.0098 at G=64) and ~5-20x at 10-100 bodies (0.0064 vs 0.033 best). The gap did NOT narrow with N; it widened (A best 4x, B best 5x but typical 20x). Scatter variants beat const-vel by 17x (A) / 8x (B) at step 20.
2. Resolution dominates the dense models: err@20 A 0.0177 (h=1) -> 0.0098 (h=0.5) -> 0.0031 (h=0.25); self-force falls similarly (0.29/0.47/0.09, noisy). Local vs U-Net matters little at these extents; recurrence helps the local net (3.0x at A, 1.5x at B) and barely the U-Net (A@20 same, B 1.15x) -- consistent with prediction 3, and with the interpretation that the recurrent medium extends the effective receptive field over steps.
3. Momentum fix is essential for momentum (5e-8 vs 6.6e-2, 4.7 in B) and slightly better err@20 (0.0098 vs 0.0160); it does not fix self-force (a measured isolated-token force of 0.1-0.5 vs a typical 2-body acceleration of ~0.1-0.4 is large). Caveat: the mass-weighted mean-dv removal is crude: in the star+light-body probe it pushes the star's spurious self-force onto light bodies as a uniform acceleration (visible as a uniform field in `scatter_field_force_field.png`, which is a probe artifact of momfix, not only the net).
4. v2 multilevel (all of the coordinator's decisions) is worse than the dense v1 on every err@k metric, by 3-10x at the same budget, and unstable at long horizon with dm applied. Neither the adaptive refinement nor the quadrupole closes the gap: ml_noquad is only marginally worse than ml_rec at err@k (A 0.060 vs 0.050) but far less stable. k_leaf=4 is much worse (a 2-body pair sits in one leaf; sub-leaf structure is lost). Likely causes (hypotheses, not isolated): (i) NGP + linear Taylor in a cell cannot represent the 1/r near-field of a neighbour in the adjacent cell; (ii) leaf-level switching as bodies move gives force discontinuities; (iii) the dm channel (kept as a trainable mass update per the decision) feeds back on itself: with dm not applied (ml_nodm) momentum drops to 2.5e-8 and the blow-up disappears, but accuracy is unchanged (A@20 0.054 vs 0.050), so dm explains the instability and momentum failure but not the accuracy gap. Not tested: longer training for ml (its loss was ~20-50x higher than v1 at the end, so it may be under-trained), CIC instead of NGP, a smooth blend between leaf levels.
5. Orbit stability / sweep (D, `scatter_field_separation_sweep.png`): with d/h_leaf from 0.25 to 16 (h_leaf=0.5, eps=0.5), the baseline is accurate everywhere (separation error <0.5% of d at step 100). Dense v1 (h=0.5 grid) holds err@20 ~0.004-0.01 for all d, with a bound-orbit separation error at step 100 of 5-9% of d for d/h in [0.5, 2] and 12-42% for d/h >= 4; the failure point is d/h_leaf = 0.25: separation error 55% (ss_rec) to 200% (ms_rec) and L drift 2.3-10.6x, i.e. it breaks at d < h_leaf/2 (fits "softening at h_leaf"). v2 does not have a clean break: ml_rec explodes for d/h <= 4 (unbounded), ml_noquad breaks (sep error >100%) at d/h = 0.25 and again at 4-8 where orbits drift out, ml_nodm similar. The upper side (large d) is also bad for the learned fields because separate fast/slow dynamics at long horizon accumulate error (orbit period at d=8 is >1000 steps so 100 steps is a small arc; the errors there are mostly a force-law bias).
6. Curl of the learned force field (static star + test mass; truth 0): curl/total 0.19-0.55 for dense v1 (lower for recurrent local/U-Net at B: 0.19-0.37), 0.4-0.73 for v2. Nothing enforces curl-free; the learned field is far from conservative, consistent with the energy drift above. (Probe is distorted by momfix, see 3.)
7. Delete-the-star (C, `scatter_field_delete_star.png`, lag = steps until the test-mass force falls below half its pre-deletion value; truth 0): ms_rec and ml_norec/ms_norec lag 0 at all distances (the global receptive field reacts instantly, as predicted); ss_norec lag 0 up to d=6 and None (never reacts, because the pre-deletion force was already ~0 beyond the 9-unit receptive field: ratio_before 0.0 at d>=12), i.e. no force at all outside the RF. ss_rec (local conv, recurrent): lag 0 at d=3 and lag 1 step at d>=6, with pre-deletion force close to truth at all d (0.76-1.7x) -- the one case showing the user's retarded-propagation signature, and small (1 step = dt=0.1, c_eff ~ d/(dt) = 60+). ml_rec: lag 0, one step of tail (1) at d=16,20 in this run. ss_rec_h8 (8 extra hidden channels) was unstable (lag 11 at d=3, response oscillates, dL/L 15). So the prediction (recurrent local lag ~d/RF) was only partly right: lag is ~1 step regardless of d, not growing with d; the net learned Newtonian instantaneous behaviour as far as its receptive field allows and the recurrent state added at most one step of stale field. Honest caveats: the pre-deletion force in the recurrent nets oscillates strongly in time when the tracers are held fixed (see plot: ss_rec swings up to ~10x truth), so the 50%-crossing lag measure is noisy for them; training never contained a deletion.
8. Self-force: nonzero for all variants (isolated-token mean |a| 0.05-2.6, typical 2-body accel 0.1-0.4); lowest for the nomomfix variant (0.009) -- the probe is run with momfix off for all, so momfix itself is not what lowers it; ms_rec at G=64/B has a large self-force (2.6).

### Anomalies / caveats
- v1 variants in Exp A/C/B were run with code before the curl/sweep/mass-drift additions, so curl, sweep and per-channel numbers exist only for B, D and the v2 runs (v1 A has no curl/sweep; D has them for ss_rec, ms_rec).
- The baseline CentralForce D numbers use the A-trained checkpoint (random 2-body pairs, not orbit data), yet it is accurate across d/h, so the comparison is if anything biased against the scatter models only mildly.
- ms_rec_g128 at A has extent 32 (h=0.25); not directly comparable per-cell with B's h=1.
- One seed (4738) per variant; differences under ~20% are within noise (see feedback_select_on_heldout_seeds).
- 4000-iteration budget for all; ml losses had not plateaued as tightly as v1 (probably under-trained, untested).
- The dm channel as specified is destabilising when applied; ml_nodm kept for attribution only.
