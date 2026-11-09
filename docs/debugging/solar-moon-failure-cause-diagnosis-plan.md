# Moon failure cause diagnosis: plan (2026-11-02)

Registered before any run. Question: why does the scatter-field model lose moons over 2000 ticks (two-moons survival 841.6 for `solar-scale-translation-augmentation-more-moon-situations`, truth about 2000)? Candidate causes: out of distribution, grokking (late generalization), model capacity, model architecture, loss function, training schedule.

Baseline checkpoint: `solar-scale-translation-augmentation-more-moon-situations` (branch commit 070d465). Comparison checkpoints: `solar-scale-translation-augmentation-retrain` (743.6), `solar-scale-same-horizon-control`.

## Loss as it is now (scripts/train_scatter_solar_stability.py:116)

Window of k ticks from stored truth (k ramps 4 to 30 over 6000 iterations). Roll the model k steps from the true start. Loss = mean squared position error + 0.1 x mean squared velocity error, averaged over all real bodies and all k steps. Every body weighs the same; error is absolute, not relative to orbit size. No energy, period or orbit-shape term. Windows are at most 30 ticks; the test is 2000.

## Evaluation fix (needed by every test below)

The moon eval has 20 moons from 10 scenes (95% bootstrap interval 631-1071; same checkpoint scored 0.72 and 0.756 on repeat). Generate at least 100 moons (at least 50 two-moon scenes, seeds from 4738 up, scene rule unchanged). Report survival with bootstrap 95% interval. A difference smaller than the interval half-width is no effect.

## Tests

1. `moon-eval-repeat-noise-floor`: re-run the eval on the unchanged checkpoint 3 times on the enlarged scene set. Reports the noise floor. No line; sets the width used below.
2. `moon-checkpoint-on-training-scenes`: score the checkpoint on its own training scenes (out of distribution test). Prediction if out of distribution: moon survival on training scenes at least 1.5x held-out. KILL for the hypothesis: within the interval of held-out.
3. `moon-survival-through-training`: moon survival and per-class loss (star, planets, belt, moons) at saved checkpoints through training (grokking curve). Needs periodic checkpoints; if none exist, the longer-training arm saves them. Prediction if grokking: survival still rising at the end and improving with more iterations. KILL for the hypothesis: flat over the last third of training.
4. `moon-overfit-few-scenes`: train on 4 moon scenes only until training loss plateaus, same architecture. Prediction if capacity or architecture: moon survival on those same scenes stays under 1100. If it reaches 1500 or more, capacity and architecture are not the limit on these scenes.
5. `moon-force-oracle-moon-rows-only`: swap the true force in on the moon rows only, model elsewhere (oracle, no training). If moons survive at least 1500, the moon force is the error source (architecture or loss or data); if they still fail, the loss of moons comes from the host planet's error or the momentum step. Compare with swapping on the host planet row only.
6. `moon-loss-weighted-up`: retrain from the translation-arm init, same 6000 iterations and data, moon bodies weighted 20x in the loss; control arm is the unchanged loss. Separate run (one change).
7. `moon-loss-relative-error`: retrain with position error divided by the body's orbital radius squared. Separate run.
8. `moon-staircase-body-count-curriculum` (the n-ramp): k and n rise together as a staircase. Start k=4, n = star + 8 planets + moon, no belt. Each stage: raise k a few ticks, then raise n, repeat. End k=30, n = full scene (200 belt plus moons). Moons in every stage. Same init, 6000 iterations, loss, data and translation as the moon arm.
9. `moon-longer-training-3x`: the moon arm recipe at 18000 iterations with checkpoints every 1000. Doubles as the grokking test if test 3 has no checkpoints.

## Lines (two-moons survival at tick 2000, enlarged eval set, truth about 2000)

Applies to training arms 6, 7, 8, 9. Guards: held-out planets kept at least 0.67, original solar at least 7 of 8 on seeds 9300 and 12000.
- PASS: survival at least 1100 and both guards.
- KILL: survival at most 900 (moon arm 841.6 plus the noise band), or a guard fails.
- Between: INCONCLUSIVE.
Replace 900 with 841.6 + the noise floor from test 1 if that is larger, before looking at arm results.

## Reading the outcomes

- Test 2 fails to differ and test 4 reaches 1500+: not out of distribution, capacity not the limit; look at loss and schedule (6, 7, 8).
- Test 4 under 1100: capacity or architecture; test 5 separates a moon-force fault from a host-planet fault.
- Test 3/9 rising: grokking or undertraining.
- Test 6/7/8 PASS: loss or schedule is the cause. Several passing: run the pair combined as a follow-up (separate dispatch).

## Rules for the agents

Descriptive names above are exact. Seed 4738 for new seeds. No inline Python. GPU only for compute (Polaris, one job per slot, per the registry). No idle-waiting on Polaris. Fresh independent reviewer before any result is reported. Log each in `docs/debugging/experiment-log.md` when done.
