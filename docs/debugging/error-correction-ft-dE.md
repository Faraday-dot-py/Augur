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

### fix1 results (jobs 3559 train, 3570 eval; wide probe = 24 nr100 / 12 bh300 fresh scenes; harness = scatter_regress run/compare vs baseline_v2)
| arm (one change vs ctrl) | probe nr100 dE@1000 med | probe expB err@20 | harness nr100 dE@1000 | harness bh300 dE@1000 / dE@100 | harness rows failing |
|---|---|---|---|---|---|
| base E | .0206 | .02415 | .0170 | .481 / .0391 | ref |
| ctrl4738 (lr 1e-4) | .314 | .02589 | .304 FAIL | .704 FAIL / .083 FAIL | 4 (+bh300 dL) |
| a: pen_e1 (soft dE loss lam 1, lam_L 0) | .0449 | .03211 (+33%) | .0457 FAIL | .275 / .056 FAIL | 14 (err rows +15..40%) |
| b: lr 1e-5 + warmup 100 | .275 | .02411 | .245 FAIL | .689 FAIL / .086 FAIL | 3 |
| c: heads only (kmlp, ppmlp; 8.8k params, U-Net frozen) | .0171 | .02474 (+2.4%) | .0218 PASS | .497 PASS / .0576 FAIL | 3 (bh300 dE@100, nr100 dL, bh300 dL) |
| d: EMA .999 (final / .ema) | .313 / .260 | .02590 / .02507 | .321 FAIL | .681 FAIL / .083 FAIL | 4 |

Predictions vs result: pen_e1 dE <= .03 MISS (.045), err within +10% MISS (+33%; dE-only penalty at lam 1 already buys dE with accuracy, like lam 3/3); lr1e5 dE .04-.12 MISS (.275: 10x smaller step size, same dE, so dE is not tracking step size; displacement ~1e-2 cumulative is already enough); heads >= .1 MISS (.017: near-field heads do NOT move dE, opposite of D4 residual conv result); ema .06-.2 MISS (.26-.31).
Mechanism read: dE@1000 is set by the U-Net weights (recurrent potential field). Any U-Net update, at any lr tried, moves nr100 dE@1000 from .02 to .25-.5; training only the pair/kernel MLPs leaves it at base (.017) with err@20 +2.4%. bh300 dE@100 harness FAIL for heads (+47%) is 4-scene noise: the 12-scene probe gives .0587 vs base .0592.

## Transfer to option C (job: ftde_c1): freeze the E U-Net, train only gin (geometry input conv), cmlp (ball-ball head), gvec, gscale
Same schedule as C (job 3494: 4000 c_train scenes x 80 ticks, 8000 it, batch 16, lr 1e-4, k 4->20, cap 100, init E, seed 4738) except `--train-only gin,cmlp,gvec,gscale`; control = original C.pt (same schedule, everything trained). Geometry-free harness rows (`scatter_regress.py run --contact`) and 24/12-scene probe for both. Predictions (before results):
- Cfz nr100 dE@1000 harness <= 1.25x base + 1e-3 (PASS) (P .6): geometry-free force is E's U-Net plus the cmlp/gvec/gscale terms (zero-init, trained); heads-only E run stayed at base. Original C fails it by a wide margin (P .85, unmeasured before).
- Cfz c_id err@20 <= 1.3x C (5.1 -> <= 6.6) (P .5): geometry only enters the frozen net through the first conv; capacity is limited. tunnel@20 <= 5% (P .55).
- Cfz bh300 dE@1000 harness PASS (P .5, noisy 4-scene row); harness err rows PASS (P .8).
- Runaway: off-grid fraction and err@100 median reported from the saved trajectories; no prediction of removal (P(runaway gone) .25), since the unstable mode lives in the U-Net recurrent path, which is frozen at E's contracting weights (that raises P to .4 if gin only drives a bounded geo term).

### Option-C results (jobs 3579 train+harness, 3583 runaway metrics, 3587 free-space probe)
c_id (seeds 9100-9300 x 24 scenes, 72 scenes for the runaway metrics, tunnel = center beyond surface):
| model | err@20 mean / median | err@100 mean / median | off-grid balls @100 | scenes with a ball >100 units away | tunnel@20 / @100 |
|---|---|---|---|---|---|
| C (original, everything trained; control) | 5.10 / 4.13 | 1767 / 1031 | 95% | 100% | .3% / 76% |
| Cfz (E U-Net frozen; gin, cmlp, gvec, gscale trained; 5.6k params; 12 skipped steps, all after it 6800) | 7.92 / 7.66 | 233 / 185 | 98% | 99% | 59% / 74% |
Predictions: Cfz err@20 <= 6.6 MISS (7.92, +55% vs C); tunnel@20 <= 5% MISS (59%); runaway gone MISS (off-grid 98%). The frozen net cannot form walls through its first-conv geometry input alone. Unbiased image review (fresh agent, `videos/contact_c_id_{Cref,Cfz}_grid.png`, prompt `contact-grid-review-prompt.md`): C red holds a compact cluster inside the container to ~tick 30 then errors jump and red is absent from tick 60; Cfz red scatters, leaves the container by tick 30, with balls on the obstacle at tick 10, smooth error growth, absent from tick 60. Consistent with the metrics.

Harness rows for C (`scatter_regress.py run --contact`): as-is (job 3579) C fails 27 rows, Cfz 28 (err@5 .64 vs .001), because C's local force includes the learned uniform gravity vector gvec (contact-world gravity, +x) and ball-contact head; these are not the N-body gravity force the rows measure. With gvec and cmlp zeroed (`--contact-free`, job 3587): C still fails 27 rows (nr100 dE@1000 1.9e4; the U-Net trained on contact scenes is far from E in free space), Cfz fails 21 (nr100 dE@1000 .86, bh300 dE@1000 .095 PASS, err@5 .12) because gscale drifted to 1.28 (scale on the base force); with gscale = 1 Cfz's free-space force is exactly E (frozen net, zero geometry input). Probe (24/12 scenes) nr100 dE@1000: C 9.9e3, Cfz .94 (gscale 1.28), D3b control 8.7.
Flag: the "every C fine-tune fails nr100/bh300 dE" row is largely a category error. C is a different physical model (contact world with its own gravity and force scale); the gravity-harness dE rows are not defined for it unless the free-space part is pinned. The E-testbed result stands on its own.

## Verdict
- Diagnosis (E testbed, 3 seeds of evidence: ctrl 4738/4739 + 3 other arms): any update of the U-Net weights moves nr100 dE@1000 from .02 to .2-.5 within ~25 iterations at lr 1e-4, and at lr 1e-5 over 2000 it; interpolation back toward E scales dE ~linearly (a=.25 already 3x base); EMA does not help; a soft dE loss recovers dE only at a 30% accuracy cost; training only the near-field heads (kmlp, ppmlp) keeps dE at base (.0218 harness, PASS) with err@20 +2.4%, but gives no accuracy gain either.
- No fix found for option C. Freezing the U-Net kills dE by construction but also kills contact capability (tunnel@20 59%).
- Not tested (candidates for the next single-change run, each needs a decision): (1) separate geometry branch: small net on geometry channels producing the bounded wall term, E U-Net fully frozen, free-space force == E exactly; (2) anchor loss: soft penalty |a_model(no geometry) - a_E| on free-space scenes while the U-Net trains (soft prior, no hard-coded force); (3) soft dE loss at lam 0.1-0.3 on C (lam 1 already costs 30% on E).
