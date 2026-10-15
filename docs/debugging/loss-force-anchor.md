# Force-anchored auxiliary loss, scatter field (2026-10-14, uncommitted)

Branch/worktree `loss-force` (`.claude/worktrees/loss-force`, from `loss-tests` = master f0e6238 + loss-tests' uncommitted edits copied in). Changes: `scripts/train_scatter_field.py` (loss-tests' `rloss` + optional `args.force_aux` callable added to the loss), new `scripts/scatter_forceaux.py`, `scatter_forceaux_summarize.py`; `scatter_error_growth.py` / `scatter_rootcause.py` copied from master's untracked files for the force metric.

## Design
- Base = Euclid rollout loss exactly as loss-tests (variant `ms_kp_pot_v_g128`, pp=2, knn=16; Exp E data 2000 scenes x 30 ticks seed 4738; B (batch) = 16; k (unroll steps) 4->20; cosine lr; 4000 it; loss = lp(Euclid) + 0.1 lv + lm). Control = this loss alone.
- Single change: + L_aux. Per training window, run the model's force teacher-forced along the k+1 truth states (recurrent field carried along truth, as in the force metric), and at ticks 2..k take, per body, e_i = |a_model,i - a_ref,i| / |a_exact,i| (relative, not squared: robust). L_aux = lambda x mean of e_i over selected bodies.
  - Selection rule (fixed up front): nn < 1 (nn = distance to nearest neighbour) and |x|_inf < 26 (inside the metric's interior box). Reason: 95% of E's squared force error is at nn<1 (long-rollout-root-cause.md); far bodies are left to the rollout loss.
  - lambda = 0.32 (fixed, not swept) = measured slope of err@20 per unit iid relative force noise on expB (exact Verlet + 8% noise -> err@20 .0256, Results A of long-rollout-root-cause.md), so L_aux is in the same units as the position loss (the tick-20 position error that force error would cause).
  - Arm A (exact): a_ref = exact fp64 softened (eps .5) pairwise force on the same (float32) state. Training label only; inference is unchanged.
  - Arm B (anchor): a_ref = frozen Exp B ckpt's force on the same states with its own carried field; applied on the 1000 scale-init scenes only (B is in-distribution there; B has 20.6% force error on clusters, so anchoring there would pull toward a wrong force). Warm setup only: the anchor's purpose is to stop the warm start from leaving B; in scratch it would be a teacher-distillation experiment, not this one. Run as a cheap parallel arm in the same job because the first-step degradation is a departure from B, which only an anchor targets directly.
- Setups: W (warm, primary) = init Exp B ckpt, fresh Adam, lr 5e-4; S (scratch) = random init seed 4738, lr 1e-3.
- Force metric (reused exactly from `scatter_rootcause.force_analysis`, no shifts): rel RMS = sqrt(sum |a - a_exact|^2 / sum |a_exact|^2) over bodies with |x|_inf < 26, ticks 2-99, field carried along truth; expB 24 scenes seed 9300, cluster 12 seed 9200 (B ckpt read 0.94% / 20.6%, E 6.1% / 6.5%). Measured at iterations 0, 1, 10, 100, 4000 (= number of Adam updates applied), by nn bin <0.5, 0.5-1, >=1.
- Eval: seeds 9100/9200/9300, 24 `ab` (10-200-body scale-init, expB-like) + 24 `cl` (cluster) scenes x 100 ticks each: err@5/10/20/50/100, step-20 per-body error median/mean/p99; `scatter_regress.py` run/compare vs `scatter_baseline.json` incl. its conservation rows nr100/bh300 dE@1000 (`scatter_conservation.py --per-seed` does not exist on this branch; the regress harness calls the same conservation code).
- Kill criterion: force rel RMS at nn<1 lower than the Euclid control AND ab err@100 better than control by >10%; otherwise negative. <10% single-seed differences = noise. Clear effect -> replicate seed 4739.

## Predictions (written before any run)
- Iteration 1 (all warm arms): the fresh-Adam first step is a sign-like update of size ~lr on every weight, insensitive to loss scale, so the jump 0.94% -> ~5% expB force occurs in all three warm arms (70% confidence). The aux term changes only the direction of that step.
- W-ctl (Euclid): reproduces loss-tests (ab err@20 ~.0063, @100 ~.74). Force expB 3-6% at it 4000, nn<0.5 the largest bin.
- W-A (exact): force recovers over training: expB force at it 4000 1.5-3% (vs ctl 3-6%), nn<1 bin 40-60% below ctl; ab err@20 20-40% below ctl, err@100 15-30% below (55% confidence of passing the kill criterion). Cluster force improves more than expB in absolute terms (ctl ~5-8%).
- W-B (anchor): expB force ends nearest to B (1-2%), cluster force worse than W-A (anchor carries no cluster information); ab err@20 similar to W-A within noise; cl err no better than ctl.
- S-ctl: ab err@20 ~.0116, @100 ~1.0, force ~4-6%.
- S-A: force nn<1 30-50% below S-ctl; err@100 10-25% below (45% confidence of passing the kill criterion; scratch has less force accuracy to protect).
- Harness: dE rows fail in all arms as before; aux arms not worse than control on nr100 dE@1000 (low confidence, 50%).

## Results, seed 4738 (job 3624; raw Polaris `~/lf_results.tar`, log `~/lf-main-3624.log`)
Metric check: B ckpt (it 0 of warm arms) reads expB 0.942% / cluster 20.6%, = the 0.94% / 20.6% of long-rollout-root-cause.md (definition reproduced).
Train time 4000 it: ctl 540 s, +exact 1057 s, +anchor 1207 s (teacher-forced pass doubles force evals in training; inference cost unchanged).

Force rel RMS (expB all / nn<0.5 / nn 0.5-1 / nn>=1 | cluster all), by number of Adam updates:
| arm | it 0 | it 1 | it 10 | it 100 | it 4000 |
|---|---|---|---|---|---|
| W ctl (Euclid) | .94% | 5.58 / 5.03 / 5.58 / 6.76 \| 17.8 | 3.49 \| 15.2 | 3.67 \| 8.5 | 1.41 / 1.40 / 1.46 / 1.29 \| 6.19 |
| W +exact (A) | .94% | 4.25 / 4.13 / 4.11 / 4.83 \| 18.0 | 3.61 \| 15.7 | 1.47 \| 10.1 | 0.85 / 0.77 / 0.90 / 0.94 \| 6.23 |
| W +anchor (B) | .94% | 5.58 (= ctl) \| 17.8 | 4.75 \| 15.2 | 1.37 \| 12.5 | 0.86 / 0.82 / 0.87 / 0.92 \| 7.68 |
| S ctl | 100% | 95.7 | 75.9 | 13.8 | 1.35 / 1.32 / 1.33 / 1.46 \| 7.04 |
| S +exact | 100% | 95.7 | 79.0 | 16.2 | 1.18 / 1.05 / 1.18 / 1.44 \| 6.82 |

Rollouts, eval seeds 9100/9200/9300 (mean), ab = 10-200-body scale-init, cl = clusters:
| arm | ab err@5 / 10 / 20 / 50 / 100 | ab e20 median / mean / p99 | cl err@20 / @100 | harness expB err@20 / @100 | nr100 / bh300 dE@1000 |
|---|---|---|---|---|---|
| W ctl | .00081 / .0021 / .0063 / .056 / .721 | .0049 / .0066 / .028 | .0363 / 4.50 | .0061 / .655 | 1.35 / 2.07 |
| W +exact | .00074 / .0017 / .0049 / .044 / .591 | .0039 / .0052 / .023 | .0346 / 4.46 | .0047 / .482 | 1.91 / 2.13 |
| W +anchor | .00070 / .0017 / .0053 / .046 / .579 | .0041 / .0063 / .037 | .0365 / 4.48 | .0041 / .447 | 1.72 / 3.82 (bh300 outgrid .97, bound .14) |
| S ctl | .00104 / .0029 / .0090 / .064 / .809 | .0076 / .0091 / .035 | .0412 / 4.47 | .0090 / .753 | .89 / 1.61 |
| S +exact | .00118 / .0036 / .0125 / .086 / .903 | .0127 / .0139 / .039 | .0454 / 4.58 | .0105 / .797 | 1.03 / 2.41 |
Harness: all 5 arms OVERALL FAIL on the same rows as every earlier arm (bh300 dE/dL@100); accuracy rows pass. Truth dE@1000: nr100 4e-5, bh300 3e-4.
Run-to-run noise: S ctl here is the same config as loss-tests' S euclid (ab err@20 .0116, rep .0110) but reads .0090 (GPU non-determinism; -22%); W ctl reproduces loss-tests' W euclid exactly (.0063 / .735 vs .0063 / .721). Scratch single runs therefore carry >=20% noise.
