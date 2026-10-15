# Rollout loss-function tests, scatter field (2026-10-14, uncommitted)

Branch/worktree `loss-tests` (`.claude/worktrees/loss-tests`, from master f0e6238; uncommitted edits: `scripts/train_scatter_field.py` adds `args.rloss` in {sq (default, unchanged), euclid, huber}; new `scripts/scatter_losstest.py`). Runs on Polaris from a clean tar export in `~/augur-lt`.

Hypothesis (untested before this doc): squared rollout loss over-weights chaotic close-encounter tail bodies, so training trades bulk (median) accuracy for tail hedging; a robust position loss aligned with the mean-Euclid metric stops that.

## Design
- Variant `ms_kp_pot_v_g128` (pp=2, knn=16), loop identical to `scatter_prior_pp8.py`: Exp E data 2000 scenes x 30 ticks seed 4738, B (batch) 16, k (unroll steps) 4->20, cosine lr, 4000 it, eager, loss = lp + 0.1 lv + lm.
- Only change per arm: the position term lp. Velocity (squared, 0.1) and mass terms identical in all arms. Loss normalisation: lp averaged over steps, bodies; squared = mean over 2 components of e^2 = d^2/2 with d = per-body Euclid error. euclid: d/sqrt(2) (rms component). huber: d^2/2 for d<delta else delta(d-delta/2), so it equals the squared loss in the bulk. Gradient clip 1.0 unchanged (it binds harder for euclid, whose gradient is ~1/d larger; Adam is scale-invariant otherwise).
- delta (huber): median per-body Euclid error pooled over steps 1..20 of the Exp B checkpoint (the accurate warm-start base) on 96 training scenes, windows t0=0,5,10, cold field. One value for both setups; no sweep.
- Setups: (S) scratch: random init seed 4738, lr 1e-3. (W) warm: init = Exp B ckpt (`init_B_ms_kp_pot_v_g128.pt`, E's recipe source: 0.94% force error), fresh Adam, lr 5e-4 (E's lr), same loop otherwise. W is the "accurate base" setup where the degradation was seen. Caveat: E's it0 degradation came from the first fresh-Adam step (sign-like, lr-sized update), which a loss change cannot remove; W may therefore show a loss-independent shock.
- Eval: seeds 9100/9200/9300 fresh (never used for selection), 48 scenes x 100 ticks each = 24 `ab` (10-200 bodies, scale-init, expB-like) + 24 `cl` (clusters). err@5/10/20/50/100; per-body step-20 Euclid error mean/median/p90/p99 and squared-loss top-1% share; force rel RMS at truth states (ticks 0,10,20,30) overall and by nn (<0.5, 0.5-1, >=1); `scatter_regress.py` run/compare vs `scatter_baseline.json` for nr100/bh300 dE.
- Noise rule: single seed per arm; arm differences < ~10% are noise; a clear winner gets one replicate seed vs control.

## Predictions (written before any run)
S-ctl: reproduces the earlier ctl: ab err@20 ~.028, @100 ~1.5, force ~5%, median step-20 error ~.04.
S-euclid (A): median step-20 error 25-50% below ctl, mean Euclid error 10-25% below, force error <=4% (vs 5%), err@20 ab 15-30% below; p99 up 0-20%. Confidence moderate (55% that ab err@20 differs from ctl by >10% in the predicted direction).
S-huber (B): between ctl and A (median 10-30% below ctl).
W-ctl: force rises from 0.94% to 4-7% in 4000 it, median step-20 error up >=2x vs untrained B, held-out squared loss down.
W-euclid: shock from the first Adam step still occurs; ends with force <=3.5% and median step-20 error >=25% below W-ctl. Probability that the robust loss removes most of the median-vs-loss divergence: 40% (the Adam first-step shock and the squared velocity term are confounds).
W-huber: between W-ctl and W-euclid.
Harness: no arm passes nr100 dE@1000 (all trained away from B-on-E data); robust-loss arms not systematically worse in dE.

## Results
Jobs 3608 (smoke), 3612 (6 arms + B0 eval + regress), 3618 (replicate seed 4739, euclid vs sq, both setups). Scripts `scripts/scatter_losstest.py`, `scatter_losstest_summarize.py` (uncommitted, branch loss-tests). Raw: Polaris `~/lt_results.tar`, `~/lt_rep.tar`. delta (huber) = 0.00286 = median per-body Euclid error of B ckpt over steps 1..20 on training scenes (p90 .034, p99 .82).
Means over eval seeds 9100/9200/9300, 24 scenes each family. err@5/10/20/50/100; e20 = per-body step-20 Euclid error. B0 = untrained warm-start B ckpt (eval only). Rep = seed 4739.

ab (10-200 bodies, scale-init):
| arm | err@20 | err@50 | err@100 | e20 median / mean / p90 / p99 | held-out sq loss |
|---|---|---|---|---|---|
| B0 (init) | .0059 | .042 | .526 | .0043 / .0075 / .0174 / .0503 | 7.6e-5 |
| S sq (ctl) | .0262 (rep .0250) | .179 | 1.476 (1.426) | .0238 / .0279 / .0515 / .0943 | 5.8e-4 |
| S euclid | .0116 (rep .0110) | .093 | 1.028 (.838) | .0094 / .0118 / .0230 / .0454 | 1.1e-4 |
| S huber | .0266 | .176 | 1.545 | .0221 / .0277 / .0546 / .1001 | 6.1e-4 |
| W sq (ctl) | .0245 (rep .0202) | .175 | 1.479 (1.364) | .0218 / .0276 / .0549 / .0998 | 6.0e-4 |
| W euclid | .0063 (rep .0067) | .062 | .735 (.742) | .0049 / .0065 / .0132 / .0282 | 3.7e-5 |
| W huber | .0246 | .171 | 1.499 | .0202 / .0272 / .0563 / .1152 | 6.4e-4 |

cl (clusters): err@20 / err@100 / e20 median / mean: B0 .094 / 4.72 / .0154 / .114; S sq .0683 / 4.63 / .0517 / .0741; S euclid .0454 / 4.52 / .0270 / .0502 (rep .0458 / 4.51); S huber .0736 / 4.73 / .0553 / .0793; W sq .0629 / 4.62 / .0506 / .0693 (rep .0594 / 4.68); W euclid .0369 / 4.41 / .0180 / .0421 (rep .0357 / 4.36); W huber .0659 / 4.73 / .0501 / .0722. cl err@100 differences are 2-5%: noise (chaos ceiling).

Force rel RMS (this runner's definition: all bodies, ticks 0/10/20/30, no |x|<26 mask, so NOT comparable to the .94%/6.1% figures of long-rollout-root-cause.md; B0 reads 10.5% here): nn<0.5 on ab: S sq .055 / euclid .020 (rep .061 / .027); W sq .084 / euclid .086 (rep .091 / .085; B0 .105). The nn>=1 bin and 'all' are dominated by far bodies with tiny exact force and swing 0.03-0.97 between seeds: unreliable, not used. Force error by this metric does not explain the W-euclid gain (equal to W-sq at nn<0.5) -> improvement is not visible as lower per-tick force error at the sampled truth states.

Harness (`scatter_regress.py`, single run): every arm OVERALL FAIL on the same rows as the earlier ctl (bh300 dE/dL@100). Accuracy rows pass; expB err@20 vs baseline E: S euclid -50%, W euclid -73%, S sq +5%, W sq -11%, huber +10% / -5%. nr100 dE@1000 (E .017, truth 4e-5): S sq .125, euclid 2.12, huber .68; W sq .39, euclid 1.77, huber .078. bh300 dE@1000: S 1.11 / 2.29 / 1.50; W 1.23 / 1.98 / .85. Euclid is worst on conservation in both setups (single run, replicate not harness-checked), although its W err@100 on expB is .67 vs E 1.42.

## Predictions vs outcome
- S-ctl reproduced (ab err@20 .0262 vs predicted ~.028, @100 1.48): yes.
- S-euclid: predicted err@20 15-30% below; got -56% (twice). Median -60% (predicted 25-50%). Under-predicted. p99 went DOWN (-52%), not up 0-20%.
- S-huber: predicted between; got = ctl within noise. Prediction missed (see confound).
- W-ctl: predicted force up to 4-7% and median up >=2x vs B0, held-out sq loss down. Median up 5x (.0043 -> .0218) and mean 3.7x, but held-out sq loss ROSE 8x (7.6e-5 -> 6.0e-4), not down: here training raised the very metric it optimises on fresh scenes (contrast with pairfull's -12%).
- W-euclid: predicted shock persists and median >=25% below W-ctl; got median .0049 vs .0218 (-78%), indistinguishable from B0 (.0043) while mean (.0065) and p99 (.028) beat B0 (.0075, .050). Confidence said 40% for removing the divergence; it did.
- Harness: predicted robust arms not systematically worse on dE; got euclid worst on nr100/bh300 dE@1000 (single run).

## Verdict
Hypothesis CONFIRMED for the position-error part, in both setups, replicated on seed 4739 (ab err@20 -56%/-56% scratch, -74%/-67% warm; ab err@100 -30%/-41% scratch, -50%/-46% warm; cl err@20 -34%/-32% and -41%/-40%). Median-vs-loss divergence disappears: warm euclid keeps the B median (.0049 vs .0043) instead of 5x worse (.0218). Fresh-seed sq loss is also lower under the Euclid loss (3.7e-5 vs 6.0e-4), so the squared-loss training run was over-fitting outliers/noise rather than only trading tail for bulk; the 'top-1% share' is unchanged (.12-.25 ab) so tail concentration in the metric is a property of the data, not removed.
Huber at delta = .0029 is NOT a valid test of 'Huber': beyond delta it is delta*|d|, so the position term is ~350x smaller than Euclid-normalised and the unchanged 0.1*lv + lm terms dominate the gradient. It matched ctl (noise) and is inconclusive. A Huber with the position term rescaled by 1/delta (unit slope) would be the clean version; not run.
Not fixed: conservation (dE rows fail for all arms; euclid worse), cluster err@100 (chaos ceiling), force error at nn<0.5 in warm.

## Untested
Huber with unit-slope scaling; replicate seed harness/conservation rows; Euclid loss on the pairfull base (this is the root-cause doc's candidate #2: the analytic base start would likely give err@20 near .005); longer training / k_end; whether grad clip 1.0 binding differs between arms (not logged); Euclid loss with energy rows as the gate; Euclid on the contact/C trainer; large N; E-scale (60k it, mixed data) reproduction; force-metric definition with |x|<26 mask.

## Recommended next single change
Apply the per-body Euclid position loss to the pairfull base (analytic P3M pp=8 untruncated, zero-init residuals) vs its squared-loss pairfull control, same schedule (this is already candidate #2 in long-rollout-root-cause.md; pp=8 was dropped by user decision, so equivalently: Euclid loss on the pp=2 E recipe at full 60k scale vs E). Given the dE regression, gate on nr100/bh300 dE@1000.
