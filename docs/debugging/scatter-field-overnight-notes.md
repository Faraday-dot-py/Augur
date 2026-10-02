# Scatter-field overnight notes (live; prune stale)

Decisions: scatter-field-design-decisions.md. Baseline numbers: scatter-field-architecture.md (Results). Rules: data-gated changes, log each, follow trends, no sweeps as main strategy, dm hard-zero only if needed.
Exp A (2 body, seed 9000) err@5/10/20 | err@100 | self-f | curl: CentralForce .0001/.0003/.0007 | .009; ms_rec G64 .0010/.0019/.0098 | .229 | .47 | ~.3; ms_rec G128 .0005/.0009/.0031 | .091 | .09 (old ckpt, incompatible with current code).
Remote: submit via script_or_code; cp files to ~/polaris-mcp-files/bounce/scripts/ THEN upload sequentially (parallel cp+upload raced once). Results in remote ~/bounce/results/scatter_field/. Probe: scripts/sf_force_probe.py (rel force error vs separation).

## Change log
- iter1 (3211) potential=True: accuracy ~unchanged, self-force 6x lower; ss_pot blew up. Not adopted.
- iter2 (3212) ms_pot_g128 worse than ms_rec_g128; 60-step training windows worse. Both rejected (potential, horizon not levers).
- iter3 (3214) kernel=True (isotropic learned K(r) FFT conv, zero-init): ms_ker G64 .0007/.0017/.0058 | @100 .142 (vs ms_rec .0010/.0019/.0098 | .229); G128 .0003/.0008/.0033 | @100 .086 (vs .0005/.0009/.0031 | .091). seed 12000 mixed (G64 .0055 vs .0040). Modest win at G64, ~neutral G128. Kept as base (far-field rel force err d=4: .30->.03).
- probe (3218) rel force err vs separation d (G128 ms_ker): d=.25 .12, .5 .043, 1 .014, 2 .012, 4 .008, 8 .029. Near field (d<=0.5, force is largest) dominates remaining error.
- iter4 (job below): pp=2.0 learned antisymmetric pairwise near-field (MLP g(r), smooth window, dense masked pairs prototype; O(N) needs cell list later). Deviates from "option D skipped" (user never rejected; data says near field is the error). Flag in morning report.
- iter4 result (3221): ms_ker_pp G64 .0006/.0015/.0053 | @100 .095 (ms_ker .0058/.142), seed12000 @20 .0038 (vs .0055); G128 .0004/.0010/.0037 | @100 .099 (no gain vs ms_ker_g128 .0033/.086). Near-field rel err d=.25: G64 .057 (was .142), G128 .103 (.117). Small gain at G64 only; near-field still ~10% at d=.25. Hypothesis: unet + mesh near-field contaminate the exact PP term.
- iter5 (3222): split=True (mesh kernel zeroed inside r<pp via window, so mesh = long-range only) with unet (ms_kp_split) and without unet (kp_nonet, ablation: is the conv net helping at all for gravity?).
- iter5 result (3222): ms_kp_split .0004/.0015/.0060 | @100 .110 (no gain vs ms_ker_pp). kp_nonet (kernel+PP, NO conv net) .0010/.0035/.0153 | @100 .286, self-force 0, force err d=14 4.5% vs 65-300% for net variants => conv net adds ~3e-3 abs far-field noise; but near field (d<=1) worse without net (6%) => PP MLP under-trained there.
- Training data (init_bodies uniform +-5) has ~1% pairs at d<0.5: near field barely trained. Eval is the same distribution so mid-range d=2-8 error (abs ~5e-4..1e-3 accel) drives err@20.
- iter6 (3223): (a) ms_kp_pot = split + potential unet (pot cut self-force 6x in iter1 => less noise); (b) ms_kp_split with --orbit-mix 0.3 (30% orbit scenes d=.125..8 added to training) -> results/scatter_field_mix.
- iter6 result (3223): ms_kp_pot G64 .0006/.0014/.0049 | @100 .106, self-force .007 (vs .24), dE .128, dL .059 (best conservation so far). Orbit-mix 0.3 on ms_kp_split: .0005/.0014/.0050 (12000: .0057) => no gain; near-field data coverage not the lever. Plateau: all G64 variants err@20 .0049-.0060, @100 ~.1; G128 ~.0033-.0037.
- iter7 (3225): convergence check ms_kp_pot at 12000 iters (-> results/scatter_field_long) + ms_kp_pot_g128. Loss at end is noisy 1e-6..1e-4; if 12k helps, training-limited (consider force-weighted first-step loss).
