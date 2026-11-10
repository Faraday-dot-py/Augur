# rotation-steerable-unet-weights (prereg, diagnostic only)

Status: only the diagnostic (stage 1, spec 10a item 4) is run. The steerable retrain is NOT built and NOT authorised.

## Spec line (copied verbatim)

"Line (the core-drift line of experiment-log.md:3620, unchanged, plus a cost clause): seeds 12100/12200/12300; gates are core drift at most 0.007 per tick, err@20 at most 1.05x stock, black-hole-scene dE@1000 at most 1.15x stock. PASS all gates on every seed AND pooled dE ratio at most 1.095 AND cost at most 1.3x. KILL any value above 1.5x its gate. Fails pass: drift 0.009 on one seed."

That line applies to the retrain, which is not run. Source line at experiment-log.md:3620 (scatter-core-drift-arm-a-fresh-held-out-seeds).

## Diagnostic (what is run)

Frozen four-pass symmetrised model (legacy arm name S1c4, on ~/edge_ck/ghost_lsq.pt) vs the stock arm (legacy: S1), seed 12300, n=20000, 1000 steps, probe ticks 100/200/300/500/1000; class core15 (bodies within 15 units of the arm's own centre, raw force, momentum fix off). scripts/core_force_component_verdict.py switches the U-Net, FFT kernel and pair term off in turn (mean removed) and reports which component carries the net core force. Inputs: ~/augur_armA_fresh/results/armA_fresh/n2e4_s12300/{truth.npz,truth.json}; control stock arm is the same run. Job script: scripts/polaris_shift-average-and-core-force-diagnostics.sh (shared with the shift-average stage 1; 1.5 h limit, 1 GPU). Seed set: 12300 only (a diagnostic of one existing failure; 9100/9200/9300 do not apply).

## Diagnostic line (PROPOSED-BY-WORKER, derived from the core-drift line)

Parameters in code: CARRY 0.5, FLOOR 0.10, STOCK_X 3.0, MAJ 2 of ticks 100/200/300, MIN_BODIES 100.
- STOCK CHECK: median over the three ticks of |stock net force error| at least 3.0x |exact mean| (so the drift exists to be explained); otherwise INCONCLUSIVE.
- PASS (steerable U-Net retrain is worth building): class UNET-CARRIES in at least 2 of 3 ticks, i.e. the U-Net mean, projected on the error direction, is at least 0.5 of |error|.
- KILL (steerable retrain not justified): KERNEL-CARRIES or PAIR-CARRIES in at least 2 of 3 ticks.
- Anything else (SHARED, CANCELLATION, NO-NET-FORCE, INCONCLUSIVE): no decision, goes to the user.
Value that would still fail PASS: UNET-CARRIES in only 1 of 3 ticks, or a U-Net projection of 0.49 |error| at all three ticks. Tie-break: exactly 0.5 |error| carries.
This is a diagnostic: its result is a class, reported as such. It does not count as a PASS of the retrain line above.

## Prediction

Stated: UNET-CARRIES most likely, because the symmetrisation acts on the U-Net only and the stock core drift has been attributed to the U-Net potential (symmetrised-potential history, experiment-log.md:3394); but if symmetrisation already removed the U-Net share the remainder is likely SHARED or NO-NET-FORCE. Rough guess: UNET-CARRIES 35 / SHARED-or-CANCELLATION 35 / KERNEL-or-PAIR 15 / NO-NET-FORCE 15 (worker guess).

## Constraints check

- pp=2 knn=16: untouched (diagnostic reads existing model).
- No hard-wired conservation: none added.
- O(N): diagnostic only; the retrain, if later approved, is a weight-sharing change with unchanged per-body cost.
- No retrain is built here; cost clause of 1.3x is not evaluated.
