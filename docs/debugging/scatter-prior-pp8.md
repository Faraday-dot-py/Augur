# sf_prior_pp8 landing, pair test and ablations (decision #8, 2026-10-06)

Branch `scatter-prior-pp8`. Opt-in ScatterField flags: `prior` (kernel K = -(1+kmlp)/sqrt(r^2+eps^2) x split window, zero-init; pair term = analytic mesh complement + learned residual), `pp` (exact-pair radius), `knn`. Variants in `scripts/train_scatter_field.py`: `ms_kp_pot_v_g128_prior` (fix 1 alone, pp 2/knn 16), `..._pp8` (fix 2 alone, pp 8/knn 48, learned kernel and pair term), `..._prior_pp8` (pair 1+2). Default behaviour unchanged (prior=False, knn=16). Runner `scripts/scatter_prior_pp8.py`, scale test `scripts/scatter_pp8_scale.py`, Polaris wrapper `scripts/polaris_scatter_pp8.sh` (runs from a private copy `~/augur_pp8` so other agents' `~/augur` is untouched).

Protocol: all models trained from scratch, identical loop, on Exp E data (2000 scenes x 30 steps, half 10-200 body scale-init, half 80-300 body clusters), seed 4738, batch 16, k 4->20, lr 1e-3 cosine, 4000 it. Eval: Exp E held-out sets seeds 9000/12000 (48 scenes: `ab` = 24 scenes 10-200 bodies, `cl` = 24 clusters) and the frozen regression harness (seeds 9100/9200/9300/4738: two_body, expB, bh300). CF = CFDense (all-pairs CentralForceDynamics-equivalent) in the same loop. Frozen Exp E checkpoint (60769 it, warm-started from B) as the reference row; ctl = same architecture as frozen at matched 4000 it.

## Predictions (written before the 4000-it run)
- pair: err@20 on `ab` at least 40% below ctl; on `cl` the gain is smaller or negative (knn=48 truncation inside dense cores, n within pp=8 can exceed 48). Force rel error at least 3x below ctl.
- prior alone (pp 2): no better than ctl at 4000 it (earlier finding: prior with pp=2 is worse, starts at 14% force error).
- pp8 alone: partial gain over ctl (20-40% on err@20), clearly worse than pair.
- Ordering err@20 on `ab`: pair < pp8 < ctl <~ prior. CF close to pair on `ab`; CF vs pair on `cl` undecided.
- Scalability: with knn=48 and pp=8 the truncation fraction passes ~50% of bodies near N~1000 uniform in the 57.6-unit box (about 0.3 bodies/unit^2 x 201 unit^2 = 60 neighbours within pp), so force error rises sharply from N~1000 and in the Gaussian core already at N~300.
