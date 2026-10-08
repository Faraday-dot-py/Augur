# sf_prior_pp8 landing, pair test and ablations (decision #8, 2026-10-06)

Branch `scatter-prior-pp8`. Opt-in ScatterField flags: `prior` (kernel K = -(1+kmlp)/sqrt(r^2+eps^2) x split window, zero-init; pair term = analytic mesh complement + learned residual), `pp` (exact-pair radius), `knn`. Variants in `scripts/train_scatter_field.py`: `ms_kp_pot_v_g128_prior` (fix 1 alone, pp 2/knn 16), `..._pp8` (fix 2 alone, pp 8/knn 48, learned kernel and pair term), `..._prior_pp8` (pair 1+2). Default behaviour unchanged (prior=False, knn=16). Runner `scripts/scatter_prior_pp8.py`, scale test `scripts/scatter_pp8_scale.py`, Polaris wrapper `scripts/polaris_scatter_pp8.sh` (runs from a private copy `~/augur_pp8` so other agents' `~/augur` is untouched).

Protocol: all models trained from scratch, identical loop, on Exp E data (2000 scenes x 30 steps, half 10-200 body scale-init, half 80-300 body clusters), seed 4738, batch 16, k 4->20, lr 1e-3 cosine, 4000 it. Eval: Exp E held-out sets seeds 9000/12000 (48 scenes: `ab` = 24 scenes 10-200 bodies, `cl` = 24 clusters) and the frozen regression harness (seeds 9100/9200/9300/4738: two_body, expB, bh300). CF = CFDense (all-pairs CentralForceDynamics-equivalent) in the same loop. Frozen Exp E checkpoint (60769 it, warm-started from B) as the reference row; ctl = same architecture as frozen at matched 4000 it.

## Predictions (written before the 4000-it run)
- pair: err@20 on `ab` at least 40% below ctl; on `cl` the gain is smaller or negative (knn=48 truncation inside dense cores, n within pp=8 can exceed 48). Force rel error at least 3x below ctl.
- prior alone (pp 2): no better than ctl at 4000 it (earlier finding: prior with pp=2 is worse, starts at 14% force error).
- pp8 alone: partial gain over ctl (20-40% on err@20), clearly worse than pair.
- Ordering err@20 on `ab`: pair < pp8 < ctl <~ prior. CF close to pair on `ab`; CF vs pair on `cl` undecided.
- Scalability: with knn=48 and pp=8 the truncation fraction passes ~50% of bodies near N~1000 uniform in the 57.6-unit box (about 0.3 bodies/unit^2 x 201 unit^2 = 60 neighbours within pp), so force error rises sharply from N~1000 and in the Gaussian core already at N~300.

## Results (Polaris jobs 3397 train/frozen/scale, 3401 regress; 4000 it each, seed 4738, ~7-9 min/model on H200; full-state ckpts incl. optimizer saved every log step in `train_scatter_field.train`)
Raw JSON in `results/scatter_pp8/`. Mean position error per body (sim units), Exp E held-out `ab` (10-200 bodies) / `cl` (clusters), seeds 9000+12000 averaged:

| model | ab err@20 | ab err@100 | cl err@20 | cl err@100 | harness expB err@20 / @100 | bh300 err@20 / @100 | train s |
|---|---|---|---|---|---|---|---|
| frozen E (60k it) | 0.0242 | 1.416 | 0.0525 | 4.207 | 0.0239 / 1.422 | 0.0622 / 4.774 | - |
| ctl | 0.0204 | 1.368 | 0.0490 | 4.277 | 0.0201 / 1.280 | 0.0532 / 4.859 | 540 |
| pair (prior+pp8) | 0.0352 | 1.753 | 0.0762 | 4.768 | 0.0333 / 1.581 | 0.0779 / 5.069 | 570 |
| prior (fix 1) | 0.0419 | 1.963 | 0.0878 | 4.841 | 0.0423 / 1.831 | 0.0981 / 5.171 | 553 |
| pp8 (fix 2) | 0.0243 | 1.375 | 0.0555 | 4.450 | 0.0226 / 1.238 | 0.0601 / 4.707 | 556 |
| CF (CFDense) | 0.0191 | 1.110 | 0.0432 | 4.139 | n/a (harness build_model bug for cf) | n/a | 418 |

Regression harness vs frozen E: ctl PASS, pp8 PASS, pair FAIL (expB err@10/20 +22/+39%, bh300 err@20 +25%), prior FAIL (expB +54..77%, bh300 +31..58%).

Scale (force rel error, trained pair vs trained ctl; floor = untrained pair): gauss_100 0.045 vs 0.087 (floor 0.006), gauss_1000 0.022 vs 0.051 (floor 0.33), unif_1000 0.063 vs 0.171, unif_100000 0.175 vs 0.223 (cell/adapt). Rollout err@20: gauss_1000 pair 0.758 vs ctl 0.618; unif_1000 pair 0.186 vs ctl 0.259; unif_3000 1.395 vs 1.197. Truncation (knn 48) fraction: pair 94% at gauss_1000 / 73% at unif_1000 (ctl 47% / 0%), as predicted.

## Verdict vs predictions
- pair err@20 at least 40% below ctl on `ab`: REFUTED. Pair is 73% worse than ctl (0.0352 vs 0.0204); cl also worse (+55%), not just smaller gain.
- prior alone no better than ctl: confirmed, and worse (+105% err@20).
- pp8 alone 20-40% gain: REFUTED. Neutral at 4000 it (ab +19% vs ctl at e20, harness expB +12%, bh300 +13% e20; but beats ctl on err@100 harness 1.24 vs 1.28 and two_body err@20 0.0056 vs 0.0044 worse). Roughly ctl-equivalent.
- Ordering predicted pair < pp8 < ctl <~ prior; observed CF < ctl <~ pp8 < pair < prior. CF is best everywhere on err@20/100 and trains fastest (418 s).
- Force error: untrained pair prior has 0.6% force error at N=100 (best of any), but 4000 it training raises it to 4.5% (ctl 8.7%); trained pair force error is 2x better than ctl while rollout error is worse, so force-rel-error gains do not translate to rollout under this loop.
- Attribution: the damage is in fix 1 (prior kernel; prior alone worst); fix 2 (pp8) is benign; the pair inherits fix 1's harm, partly offset by fix 2. Both fixes are individually non-beneficial for rollout at 4000 it; no synergy.
- Scalability: truncation prediction confirmed; force error rises with N as predicted for the untrained floor (0.33 at gauss_1000), trained models flatten it (0.02-0.06).
- Caveat: single seed, 4000 it only (a longer-trained prior may behave differently; frozen E at 60k it is behind a 4000-it ctl on every metric, so iteration count dominates comparisons). Initial `MODE=all` regress stage failed (MODELS comma-list bug in wrapper loop); re-run separately as job 3401. cf harness run failed (harness `model.force` assignment on an nn.Module).

## Protocol reconciliation (job 3408, results/scatter_pp8/protocol_rescore.json)
The 0.0019 (gap doc, decision #8) vs 0.0243 (this ablation) gap is NOT an eval-protocol difference: every checkpoint scores the same on Exp B eval (10-100 bodies, 48 scenes, seeds 9000/12000) and on Exp E `ab` (10-200). The difference is the TRAINING set: gap run trained on Exp B (10-100 body scale-init), ablation on Exp E (half 80-300 body clusters). The gap-run SFx `sf_prior_pp8` checkpoint loads into the landed `ms_kp_pot_v_g128_prior_pp8` class and reproduces to 4 digits (same architecture). err@20 / err@100, mean of seeds 9000+12000:

| model (train set) | eval B | eval E ab | eval E cl |
|---|---|---|---|
| CF (B) | .0026 / .289 | .0027 / .285 | .0185 / 3.70 |
| sf_prior_pp8 (B) | .0020 / .244 | .0022 / .311 | .463 / 6.54 |
| CF (E) | .0192 / 1.10 | .0191 / 1.11 | .0432 / 4.14 |
| ctl (E) | .0202 / 1.26 | .0204 / 1.37 | .0489 / 4.28 |
| pp8 (E) | .0225 / 1.27 | .0243 / 1.38 | .0554 / 4.45 |
| frozen E | .0243 / 1.37 | .0242 / 1.42 | .0524 / 4.21 |
| pair (E) | .0332 / 1.57 | .0352 / 1.75 | .0762 / 4.77 |
| prior (E) | .0427 / 1.85 | .0419 / 1.96 | .0878 / 4.85 |

Decision #8's win is in-distribution (B-trained, B/ab eval); the same bundle on cluster scenes is 25x worse than CF (.463 vs .0185). Also fixed: wrapper `regress` loop on comma lists, and `scatter_regress.build_model` cf (now `CFHarness` adapter).

## Decision 2026-10-07: what is opt-in
- `pp8` alone (variant `ms_kp_pot_v_g128_pp8`: pp 8, knn 48) is the opt-in flag; neutral vs control (Exp E ab err@20 .0243 vs ctl .0204).
- The prior kernel (`prior=True`) stays opt-in and non-default; it hurts: prior alone err@20 .0419, prior+pp8 pair .0352, vs ctl .0204 on Exp E. No prior+pp8 bundle.
- The 0.0019 for `sf_prior_pp8` was Exp B-trained and Exp B-evaluated (not comparable to the Exp E ablation); on clusters it is 25x worse than CF.
- Raw results (`results/scatter_pp8/`) are on the branch `scatter-pp8-protocol`, not on master.
