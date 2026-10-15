# Long-rollout error root cause, scatter field Exp E (2026-10-14, uncommitted)

Script: `scripts/scatter_rootcause.py` (uncommitted; runs from a clean master export `~/augur-rc` on Polaris). Seeds: expB 24 scenes seed 9300, cluster 12 scenes seed 9200 (not used in earlier diagnostics: expB 9000, cluster 9100), truth = fp64 KDK 4 substeps. Models: E = E_ms_kp_pot_v_g128; B = Exp B ckpt (same arch, E's warm-start source `init_B_ms_kp_pot_v_g128.pt`); P2/P4/P8 = analytic P3M (prior kernel -1/sqrt(r^2+eps^2) x split window, exact complement pair term, no net, all neighbours, pp radius 2/4/8). Definitions: B = batch, N = bodies, nn = distance to nearest neighbour, err@k = pooled mean per-body position error at tick k.

## Evidence from existing docs used as premises (not re-run)
- error-growth-diagnostic.md: free err curve = one-step seed amplified by chaos (Lyapunov exact); white state noise at the model's own one-step size reproduces the curve; truth-reset every 5 ticks -> ~200x lower; force error 7-10% rel at interior bodies, equal for near/far dominated; nn<1 bodies hold 60-100% of removable error; exact force + exact a_prev gives expB .0025/.014/.26 (err@20/50/100) vs model .026/.17/1.49; cluster .022/.53/3.55 vs .057/.68/4.14 (integrator floor of 1 Verlet step/tick vs 4-substep truth).
- scatter-field-fixed-budget.md: Exp B ckpt on expB err@5/10/20 .0006/.0014/.0039, err@100 .374 (E ckpt on expB: .0034@5, .026@20, 1.49@100): E, trained on a 10-200-body + cluster mix and 60k it, is ~6x worse at @20 and ~4x worse at @100 than B on B scenes.
- scatter-prior-pp8.md: untrained analytic prior pp=2 has 14% force error (pp8: 0.6% at N=100, 0.33 at gauss_1000 from knn=48 truncation); CF (all-pairs learned central force) E-trained err@20 .019 / @100 1.10 vs ctl .0204 / 1.37; trained net raises prior-pp8 force error 0.6% -> 4.5%.
- scatter-chaos-floors.md: expB reference-noise floor @20 1.1e-4, @100 .013; E is 100x above the floor at @100.

## Predictions (written before the run)
- P-A (intrinsic grid floor): analytic P2 force rel error ~10-15%, concentrated at nn 0.5-3 (mesh/PP handoff); P8 <= 1-2% in expB. P2 rollout err@20 on expB >= E (.02-.05); P8 err@20 on expB .002-.006, err@100 ~0.3 (B-like), worse than exact on cluster only through the neighbour truncation (none here: all neighbours).
- P-B (learned vs intrinsic): B on expB has rel force error ~2-3x lower than E; E's error is split ~50/50 between phase-variance (varies with sub-cell shift) and bias.
- P-C (dose response): exact Verlet-1 + iid 8% relative force noise reproduces E's err@20 within 3x only if noise acts on nn<1 bodies; far-field-only 8% noise (nn>=1) gives <= 20% of E's err.
- P-D (shift averaging): shiftavg4 lowers E's err@20/@100 by 20-40% (if phase variance is a large share of the error); the phase-only hybrid has err comparable to a fraction of E's.
- P-E (velocity/field dependence): v-dependence + field-dependence contribute < 30% of E's squared force error each.
- P-F (integrator): exact Verlet-1 err@20 .0025 expB / .022 cluster (known), exact 2-substep KDK < 30% of that.

## Final-step candidate (predictions written before run)
Branch `rootcause-pairfull` (worktree `.claude/worktrees/rootcause-pairfull`, uncommitted): variant `ms_kp_pot_v_g128_prior_pp8_full` = analytic P3M base (prior kernel + exact mesh-complement pair term) with exact-pair radius pp=8 and untruncated neighbours (knn=400 >= max N 300), zero-init learned residuals. Control = `ctl` (frozen architecture, same loop: from scratch, Exp E data seed 4738, batch 16, k 4->20, lr 1e-3 cosine, 4000 it) via `scripts/scatter_prior_pp8.py`; plus `pairfull0` (0 iterations = untrained analytic). Earlier `pair` run (knn=48, truncated) was negative; the untruncated version is the only change here.
- C-1: pairfull0 reproduces my P8 numbers (expB err@20 ~.005, cluster ~.037 on fresh seeds).
- C-2: pairfull (4000 it) err@20 on ab <= ctl x0.5 and force rel_rms <= 2% (ctl ~8.7% in earlier doc); cluster err@20 <= ctl. If training degrades pairfull0 (force rel rises above untrained) the claim "training harms an accurate analytic base" is confirmed.
- C-3: regress harness: nr100 dE@1000 and bh300 rows not worse than baseline (from scratch, not a fine-tune of E); risk: bh300/nr100 use N up to 300 so dense pair path OK; large-N O(N) cost of pp=8 untested.

## Results A: decomposition (job 3601, 24 expB scenes seed 9300, 12 cluster scenes seed 9200; raw `~/augur-rc/results/rootcause/rc1.json`, summarizer `scripts/scatter_rootcause_summarize.py`)

Free-rollout err (pooled mean per-body position error) at ticks 5 / 20 / 50 / 100:

| variant | expB | cluster |
|---|---|---|
| E (frozen baseline) | .0031 / .0254 / .169 / 1.47 | .0074 / .067 / 1.01 / 4.30 |
| B ckpt (same arch, B-data) | .0006 / .0040 / .025 / .42 | .0023 / .151 / 1.49 / 4.75 |
| analytic P3M pp=2 (no learning) | .0089 / .089 / .47 / 2.50 | .021 / .218 / 1.79 / 5.66 |
| analytic P3M pp=4 | .0020 / .023 / .127 / 1.29 | .0049 / .067 / 1.11 / 4.51 |
| analytic P3M pp=8 (all neighbours) | .0007 / .0050 / .028 / .41 | .0022 / .037 / .967 / 4.03 |
| exact force, 1 Verlet step/tick (integrator floor) | .0006 / .0023 / .013 / .253 | .0021 / .035 / .966 / 3.99 |
| exact force, KDK 2 substeps/tick | .00012 / .00045 / .0023 / .049 | .00039 / .0052 / .216 / 2.21 |
| exact + iid relative force noise 1% | .0007 / .0040 / .026 / .43 | .0024 / .039 / .98 / 4.22 |
| ... 2% | .0009 / .0070 / .048 / .67 | .0031 / .046 / 1.03 / 4.29 |
| ... 4% | .0015 / .013 / .089 / 1.02 | .0046 / .062 / 1.08 / 4.58 |
| ... 8% | .0026 / .0256 / .168 / 1.54 | .0083 / .102 / 1.38 / 4.79 |
| ... 16% | .0049 / .050 / .336 / 2.14 | .0163 / .190 / 1.89 / 5.93 |
| ... 8% on nn<1 bodies only | .0018 / .019 / .150 / 1.38 | .0066 / .090 / 1.39 / 5.06 |
| ... 8% on nn>=1 bodies only | .0018 / .014 / .087 / 1.02 | .0041 / .055 / 1.04 / 4.18 |
| E with 4-shift force average (4x cost) | .0022 / .0154 / .096 / 1.03 | .0056 / .052 / .906 / 4.05 |
| B with 4-shift average | .0005 / .0031 / .021 / .36 | .0021 / .152 / 1.52 / 4.71 |
| exact + E's phase-only deviation | .0022 / .0197 / .140 / 1.35 | .0052 / .057 / 1.12 / 4.53 |

Force error at truth states (rel_rms = rms|a_model - a_exact| / rms|a_exact|, bodies with |x|<26, ticks 2-99):

| model | expB all | expB nn<.25 / .25-.5 / .5-1 / 1-2 / 2-4 | cluster all |
|---|---|---|---|
| E | 6.1% | 9.9 / 6.5 / 5.2 / 3.3 / 2.9 % | 6.5% (nn<.25: 7.6%, carries 80% of err^2) |
| B | 0.94% | 1.0 / 0.9 / 0.9 / 1.0 / 0.8 % | 20.6% (nn<.25: 24.7%) |
| P3M pp=2 | 12.5% | 13.6 / 13.8 / 12.0 / 10.6 / 8.9 % | 7.3% |
| P3M pp=4 | 2.7% | | 2.1% |
| P3M pp=8 | 0.61% | 0.7 / 0.7 / 0.6 / 0.4 / 0.5 % | 0.50% |

Share of E's squared force error by nn bin (expB): nn<.25 31%, .25-.5 36%, .5-1 28%, 1-2 4%, >2 <1%. Bodies with nn<.25 are 5% of bodies. Sub-cell shift (8 random shifts in [0,h)^2): phase-variance share of E's squared force error 34% in expB (59% for nn>.5, 11% for nn<.25), 12% in cluster; B 31%.
Velocity dependence (force at v=0 minus force at v) as a fraction of E's squared error: 24% overall in expB (9-33% for nn<1; >100% for nn>1 where errors are tiny). Field-zeroing probe is invalid (zeroing the recurrent field is far out of distribution; ratio 12x); discard.
One-step (teacher-forced) mean position error, expB: E 6.7e-4, B 6.1e-4, exact+Verlet-1 6.1e-4; velocity E 6.9e-3, B 3.8e-3, exact+Verlet-1 3.7e-3. Cluster position: E 6.1e-3, exact+Verlet-1 6.0e-3, B 7.7e-3. (Mean one-step position error is dominated by close-encounter outliers and cannot distinguish models whose free-rollout errors differ by 11x.)

Two-body probe (isolated pair, 64 random sub-cell phases/orientations; radial bias of body 1's force in % of exact, warm field):

| r | .1 | .2 | .35 | .5 | 1 | 1.5 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E | -20 | -16 | -7 | -0.5 | -2.6 | -6.5 | -15 | -9 | +2.6 | +5 | +1 |
| B | -2 | -1 | -0.1 | -0.8 | -1.1 | -0.3 | -0.2 | +5 | +4 | -1.5 | -0.3 |
| P3M pp=2 | +16 | +16 | +16 | +15 | +13 | +10 | -53 (at r=pp exactly) | +1.8 | | | |
(E rms error beyond r~8 is high only because the exact force there is small; E at v=1.5 relative speed: 4x worse at r>=6.)

## Results B: how E's error evolved during its own training (job 3602, eval-only on saved E checkpoints; force rel_rms | err@20 | err@100 | cluster rel_rms | cluster err@20)

| ckpt | expB force | expB @20 | expB @100 | cluster force | cluster @20 |
|---|---|---|---|---|---|
| B init (job 3601) | 0.94% | .0040 | .42 | 20.6% | .151 |
| E it0 (after the first Adam step, lr 5e-4, fresh optimizer) | 5.7% | .075 | 2.61 | 17.5% | .355 |
| E it5000 | 18.8% | .049 | 2.09 | 8.2% | .136 |
| E it10000 | 6.8% | .034 | 1.78 | 5.3% | .090 |
| E it20000 | 6.5% | .037 | 1.86 | 5.6% | .102 |
| E it40000 | 7.0% | .032 | 1.65 | 6.0% | .085 |
| E it60000 (= final) | 6.1% | .026 | 1.47 | 6.5% | .069 |
Exp E warm-started from B: the first update with a fresh Adam state at lr 5e-4 (E_ms_kp_pot_v_g128_train.json args) took expB force error 0.94% -> 5.7% (err@20 x19) and 50k further iterations (lr cosine to 5e-5) never recovered it; they kept cluster error improving. E at 60k it is worse than the 4000-it from-scratch control on `ab` (.0242 vs .0204, scatter-prior-pp8.md).

## Results C: single candidate fix, branch `rootcause-pairfull` (job 3604; uncommitted worktree `.claude/worktrees/rootcause-pairfull`, seed 4738)
Change vs control: pair term radius-complete (knn 400 >= max N) at pp=8 with analytic P3M base (prior), learned residual heads zero-init. Control `ctl` = frozen architecture, same loop (2000 scenes x 30 ticks, batch 16, k 4->20, lr 1e-3 cosine, 4000 it, eager). `pairfull0` = same net, 0 iterations. Train time ctl 537 s, pairfull 761 s (dense O(N^2) pair path; large-N cost untested). Single training seed per arm.

Runner eval, seeds 9000 / 12000 (24 `ab` = 10-200-body scale-init scenes, 24 `cl` clusters each), mean of two seeds, err@5/20/100 and force rel_rms at truth states:
| arm | ab @5 / @20 / @100 | ab force | cl @5 / @20 / @100 | cl force |
|---|---|---|---|---|
| ctl | .0027 / .0286 / 1.63 | 5.2% | .0055 / .0579 / 4.39 | 5.3% |
| pairfull (4000 it) | .0017 / .0138 / 1.04 | 4.8% | .0037 / .0345 / 4.05 | 3.0% |
| pairfull0 (untrained) | .0008 / .0050 / 0.47 | 0.68% | .0017 / .0215 / 3.76 | 0.49% |
Fresh seeds (expB 9300, cluster 9200; `scatter_rootcause_iter.py`): ctl expB .0026 / .0249 / .154 / 1.42 (ticks 5/20/50/100), cluster .0062 / .071 / 1.09 / 4.73; pairfull expB .0017 / .0152 / .094 / 1.04, cluster .0045 / .048 / .964 / 4.36; pairfull0 = the P8 row of Results A. Force error by nn bin on expB: ctl 6.0 / 4.1 / 4.0 / 3.3 / 2.4 %, pairfull 5.4 / 3.3 / 2.4 / 1.9 / 1.4 %, pairfull0 0.7 / 0.7 / 0.6 / 0.4 / 0.5 % (nn<.25 ... 2-4).
Regression harness (`scatter_regress.py compare` vs `scatter_baseline.json` = frozen E): pairfull OVERALL PASS (two_body err@20 -57%, expB err@20 -41%, expB err@100 1.42 -> 1.01, bh300 err@20 -41%, bh300 dE@100 .048 -> .038, dL@100 .302 -> .231); ctl FAIL (bh300 dE@100 .048 -> .356, dL@100 .302 -> 3.13). Conservation rows (1000 ticks, printed, not gated): pairfull nr100 dE .057 / dL .047, bh300 dE .072; ctl nr100 dE 1.31, bh300 dE 2.43; frozen E (notes) nr100 dE .017, bh300 .48. So pairfull is worse than E on nr100 dE@1000 (.057 vs .017; truth 4e-5), better on bh300; by the stricter plan-doc rule (nr100@1000 dE counts as failing) it would not pass, though this is a from-scratch model, not a fine-tune.
Timing from the harness (B=48, N~94): 3.8 ms/tick vs E 4.95; bh300 0.89 vs 2.48 ms (baseline timing definitions may differ; not a clean comparison).
Predictions: C-1 confirmed (pairfull0 expB @20 .0050, cluster .0374 on fresh seeds). C-2: err@20 <= 0.5x ctl met on `ab` (0.48x), force <= 2% NOT met (4.8%); cluster err@20 <= ctl met. Training raised the force error of the analytic base 7x (0.68% -> 4.8%) and err@20 2.8x: the untrained base beats every trained arm. C-3 partly (see dE above).

## Results D: why does training degrade an accurate base? (jobs 3605-3607, eval only)
k=20 squared-position loss exactly as in the trainer (cold field, windows start t0 = 0, 5, 10; 96 train scenes from the 2000-scene training set, 96 held-out seed-9000 scenes); per-body step-20 Euclid error quantiles on held-out scenes:
| model | train-scene loss | held-out loss | held-out step-20 err mean / median / p90 / p99 | top-1% bodies' share of squared error |
|---|---|---|---|---|
| ctl | 1.14e-3 | 7.15e-3 | .068 / .044 / .128 / .408 | 0.91 |
| pairfull | 7.6e-4 | 5.7e-4 | .041 / .021 / .089 / .333 | 0.57 |
| pairfull0 | 1.05e-3 | 6.5e-4 | .032 / .0085 / .071 / .395 | 0.66 |
| E (60k it) | 9.3e-4 | 7.8e-4 | .055 / .032 / .124 / .358 | 0.37 |
- Training pairfull0 -> pairfull lowers the squared loss (-12% held-out, -27% on train scenes) and the p99 body error (-16%) while the median body error rises 2.4x, p90 +25%, mean Euclid +29%. The loss and the reported metric (mean Euclid error) point in opposite directions; the loss is carried by a few bodies (top 1% = 36-91% of squared error; these are close encounters in chaotic dynamics). A bulk-accuracy-for-tail-hedging trade is consistent with all rows; no training run with a different loss has been done, so the mechanism is a hypothesis.
- ctl (from scratch, 32 passes over 2000 scenes) has a 6x held-out/train loss gap that is a single-outlier effect (top 1% = 91% of held-out loss) while its mean/median errors are similar on train and held-out scenes (.070/.0435 vs .068/.0438): the loss gap itself is outlier noise, not overfit of the bulk. E's train/held-out ratio equals the untrained B ckpt's (0.46-0.47), so no evidence of E overfitting (but this relies on regenerating E's training scenes with current generators; unverified).

## Literature (agent `aaa3031c3d6f61295`, abstracts/snippets only; essentially nothing verified in full text)
- No source gave force-error vs trajectory-error numbers for an N-body surrogate. "Forces are not enough" (arXiv 2210.07237) and "Beyond force metrics" (2506.14850) report low force MAE not guaranteeing stable rollouts (molecular dynamics). Consistent with Augur's pp8 history, but the measurements above show force amplitude does predict E's trajectory error.
- Integration-error-aware learning: inverse modified Hamiltonians (Offen and Ober-Blobaum 2108.02492, GP), learned integration-error terms with embedded RK safeguard (2504.05493). Adaptive/per-particle timestep criteria (Power et al. 2003 and GADGET-2 astro-ph/0505010): dt ~ sqrt(eps/|a|).
- PM floor: CIC window, aliasing, anisotropy are small-scale effects (TreePM astro-ph/0212129, FastPM 1603.00476); fixes are CIC deconvolution, optimal Green's function, interlacing (nbodykit 1712.05834). Matches the measured (h/r)^2 handoff scaling; the shift-averaged 4-phase test above is an interlacing analogue.
- ML-PM correction (2311.18017) and COCA (2409.02154): learn a Fourier-space potential correction / residual relative to a reference; abstracts report generalization, no numbers.
- MP-NODE / multiple-shooting windows (2407.00568) and DySLIM (2402.04467): train on windows sized by the Lyapunov time, evaluate with statistics rather than raw late-time error.
- Not usable as is: all verification is abstract-level; treat as leads.

## Root causes, ranked (confidence = how directly measured here)
| # | cause | evidence | confidence |
|---|---|---|---|
| 1 | Per-tick acceleration error of typical bodies in the trained model. Trajectory error equals the amplitude of that error amplified by chaos; structure is secondary | exact Verlet + iid 8% relative force noise reproduces E on expB (err@20/50/100 .0256/.168/1.54 vs E .0254/.169/1.47) and dose-response is smooth (1%: .0040, 2%: .0070, 4%: .013, 16%: .050 at tick 20); E force rel_rms 6.1% (95% of its squared error from nn<1 bodies, 67% from nn<.5, which are 18% of bodies); same architecture trained on B data: 0.94% force, err@100 .42; zero-parameter P3M pp=8: 0.61%, err@100 .41 | high |
| 2 | The floor is not intrinsic to g128/CIC/FFT or the model class. Intrinsic grid error exists only at small handoff radius (analytic pp=2: 12.5%; pp=4: 2.7%; pp=8: 0.6%) and a learned model should be able to cancel it | untrained untruncated P3M pp=8 beats trained E on both families at every horizon: expB err@20 .0050 vs .0254, @100 .41 vs 1.47; cluster @20 .037 vs .067, @100 4.03 vs 4.30; B ckpt reaches 0.9% with the same net | high |
| 3 | Training degrades accurate force: squared-error rollout loss is carried by a few close-encounter bodies, so optimization trades bulk accuracy for tail hedging (plus a harmful early update when warm-starting) | pairfull0 -> pairfull: force 0.68% -> 4.8%, mean err@20 x2.8, median body err x2.4, but held-out MSE -12% and p99 -16%; top-1% bodies hold 36-91% of the loss; E it0 (first Adam step, lr 5e-4) 0.94% -> 5.7% on expB, plateau 6-7% for 50k its | medium (consistent measurements, no loss-change training run yet) |
| 4 | Integrator floor: one force eval per tick (dt .1) against a 4-substep truth | exact-force Verlet-1: expB .0023 / .013 / .253, cluster .035 / .97 / 3.99 at ticks 20/50/100; 2 substeps: .00045 / .0023 / .049 and .0052 / .22 / 2.2. E in clusters is already at the floor for ticks >= 50 and its cluster one-step position error equals the floor (6.1e-3 vs 6.0e-3). For expB it binds only once force error is below ~1% (B and P8 are 1.7x the floor) | high (measured), but whether a learned force can absorb it is untested |
| 5 | Sub-cell phase dependence of the mesh force | 34% of E's squared force error (expB), 12% cluster; 4-phase force average: E err@20 -39%, @100 -30% (cluster -22%, -6%), B -24%, -14%; E err^2@20 = phase-only^2 + averaged^2 (.0197^2 + .0154^2 vs .0254^2) | medium-high |
| 6 | Spurious velocity dependence of the force | force(v=0) vs force(v) = 24% of E's squared error in expB (9-33% for nn<1) | low-medium (upper bound) |
Not causes (established earlier or here): recurrent field (job 3600: <5%), edge band, grid-edge padding, softening mismatch; Lyapunov exponent (model equals exact).

## Untested
- Whether a robust (Euclid-norm / Huber) rollout loss prevents the degradation in #3 (the discriminating run).
- Whether the learned force absorbs integrator error (it would show as a physical-force bias at small separations; E has -20% at r=0.1, -16% at 0.2, -15% at r=2 in the pair probe, B -2% / -1% / 0).
- Large-N cost and scaling of an untruncated pp=8 pair term (neighbours within 8 grows with density; O(N) only for bounded density); behaviour at N >> 300, in the 1e9 scenes, and harness conservation at 1e4 ticks.
- Mechanism of E's it0 degradation (first Adam step) and why 50k further iterations did not recover it.
- E's training scenes reproduced exactly (overfit check relied on regeneration).
- Pair-probe anomaly of -53% at r exactly equal to pp for the analytic P2 model (and -28% / -15% for P4 / P8 at their own pp): one radius per model, probably a measure-zero artefact at the window edge, not re-probed on a finer r grid.
- Field-zeroing probe value is invalid (out-of-distribution input).
- Single training seed per arm; 12-24 scenes per family.

## Fix candidates (ranked; none beyond the first is implemented)
| # | candidate | expected effect | cost / risk | status |
|---|---|---|---|---|
| 1 | Analytic P3M base, pp=8, untruncated pair term (branch `rootcause-pairfull`) | measured vs same-schedule control: err@20 -52% (ab), -40% (cl), err@100 -36% (ab), harness PASS, bh300 dE/dL pass | pair cost ~ density x pp^2 per body (dense path here is O(N^2)); nr100 dE@1000 .057 vs E .017 | run, one seed |
| 2 | Same base, trained with a robust rollout loss (per-body Euclid norm, not squared) so training stops trading bulk for tail | upper bound = untrained base: expB err@20 .005, @100 .41 (another 3x on top of #1); risk: tail bodies get worse | no inference cost; one training run + control | proposed, not run |
| 3 | 2 substeps per tick with the above | integrator floor / 5-7x (expB @100 .25 -> .05, cluster @50 .97 -> .22) | 2x force evals per tick | only useful after #1/#2 |
| 4 | Interlaced 4-phase force averaging at inference on E (or any mesh model without an analytic base) | E err@20 -39%, @100 -30% | 4x mesh cost; earlier random per-tick jitter (D6) raised dE; deterministic interlacing untested for dE | measured on error only |
| 5 | Drop warm-start with a fresh optimizer at nominal lr; do not fine-tune E | removes the it0 degradation | none | observation only |
Decisions needed: (a) accept a larger exact-pair radius (cost scaling with local density) as the architecture, (b) approve the robust-loss run (#2) as the next single-change experiment, (c) whether the nr100 dE@1000 .057 vs .017 is tolerable for a from-scratch model, (d) 2-substep integration trade-off.
