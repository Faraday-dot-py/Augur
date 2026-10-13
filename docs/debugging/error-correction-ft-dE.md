# Fine-tune dE regression (follow-up option 6), 2026-10-13

Branch `fix-finetune-dE` (on `worktree-agent-a98bdf87385c8eb37` = option C code). Remote dir `~/augur-ftde` (code `ftde_code.tgz`). Seed 4738, gravity 9 (token-model rule; this is the scatter field, units as the harness). dE = |E(t)-E(0)|/(K0+|U0|). Testbed: E fine-tune recipe of ft_ctrl (job 3368: 2000 it, lr 1e-4, batch 16, k 20, 1500 scenes x 50 ticks; nr100@1000 dE .017 -> .29), 7 min per run; the winning fix is then repeated on the option-C trainer.

## Known before any new run
- Harness rows nr100@1000 dE and bh300 dE are medians over 4 scenes (scatter_conservation SEEDS x 1 scene): coarse, "bh300 dE varies +-30% run to run" (scatter-fixes doc).
- ft_longpen (lam_E = lam_L = 3, 5 x 20-tick windows): nr100 dE .0044, bh300 dE@1000 .097, but err@5/10/20 +29..76% (FAIL gate). ft_sub2 (2 substeps): nr100 dE .030, bh300 .12, err -20%.
- D3 E control (lr 5e-5, batch 32): nr100 dE .332, err@20 -13..-20%. D4 learned residual: .020 -> .21. D6 jitter: x5.8. All fine-tunes of E or C fail the same two rows.

## Diagnostic (job 3551) and predictions (written before the result)
Wide probe `scripts/ft_de_probe.py`: nr100 24 fresh scenes (seeds 13000+), bh300 12 scenes, 1000 ticks; median/p16/p84 of dE at 100/300/1000, expB err@5/20 (48 scenes). Specs: base E, ctrl snapshots at it 0,10,25,50,100,200,400,800,1200,1600,2000, ctrl seed 4739 (variance of the fine-tune), weight interpolation base + a(ctrl - base), a = .25/.5/.75; base on a second scene set (probe noise).
- P1: probe is not the issue: base E 24-scene median nr100 dE@1000 .02-.06 (4-scene .017 is on the low side), ctrl .15-.4 (P .65).
- P2: dE rises early, not gradually: ctrl it100 median nr100 dE@1000 >= 5x base (P .5); the jump coincides with lr peak/fresh Adam (it < 200).
- P3: seed 4739 ctrl within a factor 2 of seed 4738 (P .7) => the regression is systematic, not a bad draw.
- P4: weight interpolation is strongly nonlinear: a = .5 recovers dE to <= 3x base while keeping >= half the err@20 gain (P .4).
- P5: expB err@20 improves 10-20% for every ctrl snapshot from it 200 on.

### Diagnostic results (job 3551; E fine-tune recipe ft_ctrl, probe 24 nr100 / 12 bh300 fresh scenes; `[p16,p84]` over scenes)
| spec | nr100 dE@1000 med [p16,p84] | nr100 dE@100 | bh300 dE@1000 | expB err@20 |
|---|---|---|---|---|
| base E (scenes 13000 / 14000) | .0206 [.011,.034] / .0187 [.004,.031] | .0034 | .640 | .02415 |
| ctrl4738 it0 / it10 | .0132 / .0133 | .0036 / .0030 | .59 / .62 | .02414 / .02503 |
| it25 / it50 / it100 | .221 / .480 / .197 | .016 / .027 / .011 | .27 / .71 / .65 | .02591 / .0259 / .02497 |
| it200 / it400 / it800 | .307 / .465 / .080 | .014 / .024 / .003 | .14 / .66 / .71 | .0261 / .0252 / .0269 |
| it1200 / it1600 / final (it2000) | .453 / .316 / .314 | .022 / .015 / .016 | 1.0 / .88 / .84 | .0268 / .0258 / .0259 |
| seed 4739 it100 / it400 / final | .175 / .522 / .369 | .003 / .028 / .019 | .51 / .66 / .77 | .0267 / .0268 / .0274 |
| ctrl4738 interpolation a=.25/.5/.75 | .063 / .145 / .229 | .0027 / .0074 / .0106 | .66 / .78 / .78 | .02428 / .02464 / .02519 |

Reading: P1 hit (base median .019-.021, probe repeatable; the harness .017 is a 4-scene value in the same range). **The jump happens between it 10 and it 25** (.013 -> .22 at lr 1e-4, i.e. < 25 Adam steps, per-parameter displacement <= ~2.5e-3), then wanders .08-.48 with no trend: dE@1000 is a very sensitive, non-monotone function of the weights, and nothing in the loss pulls it back (P2 hit, but not tied to the training loss decreasing). Seed 4739 reaches the same range (.17-.52): systematic (P3 hit). Interpolating back toward E is ~linear in a (.063 at a=.25, 3x base already): a=.25 of the ctrl update already fails the row (P4 MISS: no free lunch from weight averaging; dE grows ~ linearly with step size). expB err@20 is NOT improved by the fine-tune on this wide probe (.02415 -> .0259, +7%; err@5 +7%): the "accuracy gain" of ctrl on harness rows is not visible on 48 fresh scenes (P5 MISS), so this recipe buys no accuracy and costs 15x dE. bh300 dE@1000 (12 scenes) is noisy (.14-1.0) and ~base .64 -> .84; the official bh300 4-scene row mostly fails by noise.

## Batch fix1 (job 3559): four single-change arms vs ctrl4738 (same schedule: 2000 it, batch 16, k 20, seed 4738), each probed wide + harness
Pass bar for nr100 dE@1000 on the wide probe: <= 1.25x base + 1e-3 (~.027), err@20 not > +15% vs base. Predictions (before results):
- pen_e1 (a: soft energy-drift loss, 5 x 20-tick free windows, lam_E 1, lam_L 0; ft_longpen showed lam 3/3 reaches .0044): nr100 dE@1000 <= .03 (P .6); expB err@20 within +10% of base (P .55); bh300 dE@1000 not worse than base +25% (P .4).
- lr1e5 (b: lr 1e-5, 100-it warmup): dE ~ .04-.12 given dE ~ linear in step size (P .6); passes bar (P .15).
- heads (c: only kmlp/ppmlp pair/kernel MLPs trained, U-Net frozen): dE >= .1 (P .7), as D4 residual (.020 -> .21).
- ema (d: EMA .999 of weights, evaluated as final and .ema): .06-.2 (P .6); passes (P .1).
