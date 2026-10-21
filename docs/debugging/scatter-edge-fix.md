# Scatter-field grid-edge force artifact: root cause, fix attempts, edge scan (2026-10-20)

Branch scatter-edge-fix. Model: Exp E `ms_kp_pot_v_g128` (grid 128, extent 64, box [-32,32]^2). err@k = mean position error per body at tick k (sim units); dE = relative energy drift. Held-out scene seeds 9000/12000 (never 4738 for selection). Experiment-log entry: experiment-log.md, "grid-edge force artifact". Raw logs: results/scatter_edge/. Jobs 3786-3820.

## Root cause (T1)
The spike is the zero padding in `neg_grad` applied to the FFT-kernel potential. The kernel potential is ~N/r, nonzero at the edge, so the last cell sees a fake one-sided gradient. Using the exact 2G-FFT potential one cell beyond each side (`kext`) or replicate padding (`krep`) gives band max|a| = 1.0x exact (E: 114x), sign match 1.0 (E .58), with no retraining. Padding the U-Net potential (`nrep`) has no effect; U-Net conv replicate padding 114x -> 45x; far@0.05 114x -> 101x (taper too late); far@0.25 6x. Edge cells are rarely populated in training (0.04% of body-ticks beyond 29.5) but that is not the cause.

## Training attempts (all KILLED on the edge gate)
Each vs same-schedule control. Arms: full / input+output convs / 9-weight learned ghost stencil, with closed data (control), open-domain data (clusters crossing the box both ways, truth over all bodies, loss on bodies inside the box, edge-weighted), closed + random translation; trajectory loss and force-matching loss. Best band ratio 28x (the control, with +118% err@20); open-domain full fine-tune: err@20 x5 and wrecks 1000-tick behaviour; translation +60%; control reproduces mechanism 13 (nr100 dE@1e3 .015 -> .38). The ghost stencil trained by gradient stays at 64-114x. Table: results/scatter_edge/summary_table.txt.

## Learned ghost closure fitted by least squares (best candidate)
9-weight linear 3x3 stencil giving one ghost cell beyond each side of the kernel potential (`ghost` variant, `gpad`), fitted by convex least squares to the exact kernel potential on the doubled domain (labels include mass outside the box), over open-domain scenes at 6 ticks plus clipped Gaussians; base weights frozen. Held-out residual 1.1e-3. Soft/learned, sim-agnostic, O(N), tileable (local, 3x3).

| metric | E | fixed (ghost_lsq) | gate |
|---|---|---|---|
| band max-abs a / exact | 114x | 1.0x | <=5x PASS |
| sign pattern match | .58 | 1.00 | matches PASS |
| tracer-rollout reversal (28-32, N=2e4) | 15.7% | 1.85% | <=1% (truth .5%) NOT MET |
| BH collapse N=2e4, reversal ticks 0-300 | 7.9% | 0.32% | truth 0.01% |
| tracers r>70 per 400 | 24 | 149 (76 with far@0.05) | <=2 NOT MET |
| expB / cluster err@5/10/20 | .0030/.0082/.0242 ; .0069/.0179/.0588 | .0030/.0082/.0242 ; .0069/.0179/.0588 | within 15% PASS |
| harness bh300 dE@100 / dE@1e3 | .037 / .53 | .057 / .69 | <=1.15x FAIL (noise: E rerun nr100 dE@1e3 .0067-.045, bh300 .43-.53) |
| nr100 dE@1e3 | .0067 | .011 | noisy, FAIL on 1.15x |
| nr100 outgrid@1e3 / bh300 outgrid | .15 / .27 | .01 / .17 | better |

Why tracers escape: the model gives zero force outside the box. Exact force for tracers beyond coordinate 32 gives 0/400 beyond r>70 (job 3805); far's monopole recovers half. This is the "outside the grid" mechanism 6, not the edge band; the open-domain training did not teach it (a grid net cannot output force where no grid exists). A learned-kernel coarse direct sum (`kfar`) fixes the static outside force (1%) but its rollouts blow up energy (nr100 dE@1e3 .94); killed, not diagnosed.

## Long outward-flow rollouts (BH collapse scene, seed 4738)
N=1e5, 1000 ticks (jobs 3812; video videos/scatter_edge_fix_before_after_1e5.mp4) and N=2e4 with exact truth (job 3820). Reversal fraction of outgoing bodies in band 28-32, by tick window (N=1e5, E | fixed): 0-100 5.8% | 0.55%; 100-200 12.0% | 1.35%; 200-300 13.6% | 3.8%; 300-400 14.9% | 17.0%; 400-500 14.8% | 6.0%; truth ~0%. N=2e4 over 1000 ticks: truth 0.0%, E 12.3%, fixed 8.7%. So the fix removes the early reflection (first ~300 ticks) but around tick 300-500 the compact core itself drifts out through the edge (truth keeps frac outside box at .2; both models eventually empty the box, E by t~600, fixed by t~500) and is reflected/lost: the core leaves because the model's dynamics there differ (momentum/field), not because of the edge band. 1e4-tick rollout: not completed (cancelled; the box is empty after ~700 ticks in both). The video review (fresh agent, no hypothesis) saw no visible bouncing at the red squares in either panel, a difference in timing (fixed panel empties earlier), and a thick band of particles above the top edge in the fixed panel at the core crossing.

## Edge-artifact scan
| where | result |
|---|---|
| (a) momfix beyond grid | uniform spurious a = -0.040 outside vs exact -0.16 (N=300) ; star+light: light-body force error 4.29 with momfix vs .27 without (known side effect). Found, not fixed |
| (b) far taper | far@0.05: band 114x -> 101x (artifact still delivered); far@0.25: 6x but costs err on open scenes (+20-40% err@5 for bodies near the box edge). Confirms the research doc's anomaly |
| (c) tiles | from branch results (not re-run with fix; branch diverged, 5/8 hunks fail): seam<8 force err .07 vs interior .06 (fine); outer<8 28 tiled / 42 single (warm1200) = same edge artifact. Re-run with fix: UNTESTED |
| (d) recurrent potential field | the spike is written into the carried field: ring (3 outer cells) max 4.9e3 at t0, 4.0e3 at t100, vs interior 1.3e3; with kext the ring is <=22. No further growth |
| (e) 3D scatter grid / token windowing | absent (3D is CentralForceDynamics, gridless). scatter_contact.py inherits `neg_grad` (same spike, untested) |
| (f) IC clip +-29 | display/IC effect, previously characterized; fixed model shows no new effect (not re-measured) |
| (g) fp16 positions at the edge | max force difference fp16 vs fp32 in band 0.15 (E) / .006 (fixed), negligible |

## Code
Opt-in, defaults unchanged: `kpad` ("zero" default, `kext`, `krep`), `npad` (`nrep`), variant `ms_kp_pot_v_g128_ghost` (9 extra weights, zero-init = stock E), `kfar`. Scripts: edge_probe.py, edge_eval.py, edge_pop_stats.py, train_scatter_edge.py, scatter_edge_data.py, fit_ghost.py, edge_ghost_diag.py, edge_scan.py, edge_hybrid.py, edge_band_roll.py, edge_longroll.py, edge_window_stats.py, edge_table.py, render_edge_pair.py. Checkpoint of the fitted closure: Polaris ~/edge_ck/ghost_lsq.pt.

## Decisions for the user
1. The fix as shipped is trained-by-fit (least squares on larger-domain labels), not SGD fine-tuning, because every SGD arm failed. OK to accept?
2. dE rows: bh300 dE +31..51% vs E but within E's run-to-run noise; accept or require more seeds?
3. Outside-box force (tracers r>70) needs a separate learned far-field; is that in scope?
