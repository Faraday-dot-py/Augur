# Substep experiments, stage 1 (2026-10-14, uncommitted; branch substep-exp from master f0e6238)

## Results (job 3629, 3 seeds 9100/9200/9300 for err, 1000-tick cons on nr100+bh300; log results/substep/stage1-job3629.log)
Definitions: ab = abundant-neighbour (sparse) scenes, cl = clustered scenes; u2/u4 = 2/4 uniform substeps per tick; nf2 = RESPA step (field once per tick, near-field pair term x4 at rc=0; P0 uses rc=2); vs v1 = relative to the same model's single-step rollout.

| model | scheme | cl err@20 vs v1 | ab err@20 vs v1 | nr100 dE@1000 | bh300 dE@1000 |
|---|---|---|---|---|---|
| E | v1/u2/u4/nf2 | 0 / +1% / +10% / -3% | 0 / +9% / +28% / 0 | .018 / .93 / 1.37 / .018 | .57 / .39 / .29 / .25 |
| B | v1/u2/u4/nf2 | 0 / -4% / -2% / -3% | 0 / +32% / +63% / +8% | 1.08 / .53 / .19 / .94 | 1.39 / .45 / .14 / 1.0 |
| EW | v1/u2/u4/nf2 | 0 / -9% / +3% / -14% | 0 / +48% / +91% / +3% | 1.45 / .51 / .23 / 1.47 | 1.85 / .55 / .49 / 2.47 |
| P0 (untrained analytic pp=8, diagnostic) | v1/u2/u4/nf2 | 0 / -61% / -63% / -59% | 0 / -8% / -6% / -7% | .026 / .0064 / .0066 / .0073 | .29 / .033 / .029 / .028 |

Cost (ms/tick cl, B=24): v1 3.4, u2 6.9, u4 13.7, nf2 14.4 (learned); P0 5.6/11.1/22.2/18.6.

Verdict: no trained model gains >= 20% on cl err@20 from any substep scheme (best EW nf2 -14%, B/E within +-10%); ab gets worse under uniform substeps (+9% to +91%) as predicted for models trained at dt=0.1. Stage-2 gate (>=20%) NOT triggered (matches prediction, 60% confidence). Only the untrained analytic P0 gains (cl -60%, ab -7%), so substepping fixes integrator error only when the force is accurate; learned models absorb dt-specific bias. Misses vs predictions: B ab got worse (+32% u2), not better (-10 to -25%); E u2 ab +9% (predicted +10-20%); nr100 dE for E u2 x50 (.018 -> .93). Energy under u2/u4 improves on bh300 for E/B/EW (e.g. B .14 vs 1.39) but this is not an accuracy gain. Pair counts r<2 per body: ab 1-4, cl 17-19 (knn16 cap -> 8-10), bh1000 155-170 (cap -> 11-13), bh3000 520-620 (cap -> 12-14); symmetric union <= 33.
Not done: stage 2 (not triggered). Note job 3628 (lf-rep, force-aux replication scratch_exact_s4739) is a different line: regress PASS on err rows (-54 to -81%) but FAIL on bh300 dE_rel@100 (.048 -> .365) and dL (.30 -> 3.6); tick_ms -30 to -72%.

## Predictions (mine, written before any run)
Definitions: B = batch size, N = bodies per scene, nn = nearest-neighbour distance, dt = 0.1 (one tick), M = substeps per tick, err@k = mean per-body position error at tick k, e20 = per-body step-20 Euclid error.
- P0 (pairfull0, untrained analytic P3M pp=8, DIAGNOSTIC ONLY): u2 cl err@20 .037 -> ~.015 (-55%), u4 ~.012; nf2 (pair term windowed to r<2, M=4) ~.014-.018. ab err@20 .0050 -> .0044-.0047 for all three. nr100 dE@1000 better than its own v1.
- B (Exp B ckpt): cl no gain (<10%; its cluster force error is ~20%); ab err@20 -10 to -25% under u2/u4 (integrator ~1/3 of its squared ab error); nf2 between v1 and u2. Risk: u2/u4 iterate the recurrent field M times per tick.
- E: u2 err@20 +10-20% (matches the earlier +17%), u4 worse; nr100 dE@1000 x10 or more. nf2 (field cadence unchanged): err within +-10%, dE within 2x.
- EW (Euclid-loss warm ckpt): ab err@20 -5 to -20% under u2/u4, cl <= -15%; dE worse under u2/u4 (absorbed dt-specific bias).
- Overall: no learned model gains >= 20% on cl err@20 -> stage 2 not triggered (60% confidence).
- Pair counts per body per substep (r < 2): ab ~2.4, cl ~18, bh1000 rising with collapse to >50; the knn16 cap bounds the model's own term at 16 and the symmetric union at <= ~25 mean. My symmetric pair implementation is dense (O(N^2) distance pass, like the model's dense mode for N <= 512).
