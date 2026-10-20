# Error-correction research: index and dispatch plan (2026-10-12)

Status: three of four research agents returned; empirical diagnostic pending. Once it returns, one agent per recommended option below is dispatched (single change vs frozen baseline, same-schedule control, seed 4738, fresh-seed selection, nr100@1000 dE and bh300 dE count as failing rows for any fine-tune). Nothing below is run yet.

## Writeups (all uncommitted)
- `error-correction-literature.md` — general literature survey
- `error-correction-codebase-proposals.md` — code-grounded proposals (file:line hooks)
- `error-correction-unet.md` — U-Net rollout specific
- `error-growth-diagnostic.md` — empirical diagnostic on Polaris (PENDING)

## Options to pursue (one agent each after diagnostic returns)
Eval-only / inference-only first:
1. Cold-reset recurrent-field diagnostic (unet #3; review D3)
2. Noise-amplification probe: inject noise into a0/a_prev at tick 15, measure per-tick gain (literature first step; sets noise scale for #5)
3. Edge guard band, replicate/extrapolated halo, re-probe far@0.05 (unet #2; codebase #3 free discriminator: square-wall probe on edge-population hypothesis)
4. Grid-origin jitter, per-tick sub-cell offset (codebase #1)
5. Shift-equivariance probe on frozen E; kill if <2% (codebase #2)

Training single-change runs:
6. Noise on recurrent channels in training (literature #1, unet #4, codebase #9)
7. k_end 50 curriculum with grad clipping (unet #1)
8. Recurrent-state contraction: zero-init forget gate + gain penalty; test on contact C ckpt past tick 25 (codebase #4; literature #2 Jacobian penalty)
9. Soft-bounded grid accel, tanh (codebase #5, unet #5; weak alone)
10. Edge-exposure augmentation / random translation (codebase #3)
11. Galilean-boost augmentation (codebase #6; locate velocity-dependence probe first)
12. Distributional soft loss, MMD on dE/speeds/g(r) (literature #3; non-stationarity caveat)
13. Failure-state restarts with observable loss (literature #4)
14. Soft multi-step energy/restitution loss (unet dE, codebase #8; locate contact trainer first)
15. Anti-aliased down/up path (unet #6; unverified)

Deferred / expensive: local denoiser on recurrent field (+60% step cost), PDE-Refiner (4x), time-reversal loss, Jacobian commutator penalties.

## Disagreements to resolve with the diagnostic
- Edge spike cause: zero padding (unet agent) vs poorly populated edge cells in training (codebase agent; `neg_grad` padding touches only the last cell).
- Runaway scope: only fine-tuned contact ckpts (C, A2); Exp E zero-shot stable to 1e4 ticks. Fixes judged on C/A2, then checked not to regress E.

## Diagnostic results (error-growth-diagnostic.md, jobs 3508-3510) and reprioritization
- Runaway is NOT in Exp E; reproduces only in option-C fine-tune weights (also on gravity scenes). Carried by the recurrent potential channel, x3.5 per 3 ticks. Zeroing it every 5-20 ticks or clamping to +-100 removes it (clamp +-1000 does not).
- Edge force error does not move trajectory error: edge guard band is LOW priority for accuracy (resolves disagreement: neither padding nor edge population matters for trajectories).
- E long-horizon error = chaos amplifying flat one-step error (Lyapunov equals exact). Truth-reset of pos/vel every 5 ticks gives ~200x lower err@100; field reset does nothing for E. Residual corrector cannot beat chaos; only frequent state correction.
- ~60%-100% of removable error is bodies with nearest neighbour <1 unit: near-field residual is the target.
- Star family (OOD): model amplifies perturbations x1000, exact orbits do not.

Dispatched batch 1 (supported by diagnostic): D1 bounded/soft-saturated recurrent field (eval-only on C), D2 recurrent-state contraction (gate + gain penalty), D3 recurrent noise probe + training noise, D4 near-field residual / frequent state-correction family, D5 k_end 50 curriculum w/ grad clipping, D6 shift-equivariance probe + grid-origin jitter (eval-only).
Deferred: edge guard band, edge augmentation, Galilean boost, restitution/energy losses, distributional loss, failure restarts, anti-aliasing, time-reversal, denoiser, PDE-Refiner.

## Batch 1 results (2026-10-12; detail docs are in each agent's worktree under .claude/worktrees/, uncommitted)
- D1 bounded field (job 3525): tanh/alg softcap s=100 on fed-back field removes C contact runaway (76% -> 0, err@100 1770 -> ~90-110); s=200 half, s=400 none. C err@20 +10% at s=100. C gravity runaway still 40-67%; C nr100/bh300 dE@1000 still fail. E unchanged. Verdict: cheap inference guard, not a fix.
- D2 contraction: step 1 FAILED by bug (clamp(tanh(z),0,1) zero grad at init -> gate never moved; gain penalty is 0 from tick 2, runaway not visible to local-gain probe). Not a negative on contraction. Retry needs nonzero-grad gate + large-signal-map loss (field out vs in).
- D3 recurrent noise (jobs 3517, 3533): probe: E damps field noise 0.6-0.9/tick, C amplifies 1.2-1.65/tick (linear, 1e-5 enough). E fine-tune noise 1e-2 == control, both fail nr100 dE@1000 (.017 -> .332), bh300 dE@100, vir_gap: failure is from fine-tuning itself. Runaway test on C NOT run -> launched (see below).
- D4 near-field (jobs 3523, 3529): NEGATIVE. Exact near force caps at 1.6x; truth-reset of near bodies 26-160x. Learned local residual: one-step MSE -44% but rollout null (ratio ~0.98-1.0), fails energy rows (nr100 dE .020 -> .21). Sub-stepping useless. Only state assimilation from observations would give the gain.
- D5 k_end 50: pending (job 3521 cancelled; resubmitted with 1 CPU).
- D6 shift/jitter (job 3526): E not shift-equivariant at sub-cell (0.5 cell -> 6-10% force diff, ~ model error in rollout). Jitter: err better, but nr100 dE@1000 x5.8 and bh300 dL worse -> reject as default. Phase error NOT a coherent bias (coherence 0.12-0.16 ~ random): gain is variance averaging.

## Batch 1 final additions
- D3b (job 3536): INCONCLUSIVE. Control and noise arms both 0% |a|>200 runaway but both degraded (c_id err@20 8.5/7.7 vs C 5.1; tunnel@20 31/38%; 14-17% skipped steps past k~20); mean err@100 >> median => ejected balls missed by metric. Noise effect not separable.
- D5 (k_end 50): control (k_end 20, lr 5e-5, staged, rollback) ALSO removes runaway (0%), so the lower-lr staged recipe, not the longer horizon, is what fixes it. D5 loses wall rebound (0/7 speeds vs ctrl 7/7), err@20 8.0 vs ctrl 10.3 vs C 6.45; off-grid fraction .54-.62 at t100. All harness dE rows fail for C/ctrl/D5. Not tried: gradient-norm guard (would catch it-2450 step).
- Common lesson: any fine-tune of C degrades in-horizon accuracy (err@20 +25-60%) and fails dE rows; the runaway metric must include off-grid fraction and err@100 median.

## Follow-up options
1. D3b (LAUNCHED): C-trainer run with recurrent noise sigma 1e-2 vs same-recipe control, train past tick 25; judge runaway rate, err@20/@100, harness rows.
2. D2b: redo contraction with nonzero-grad gate (e.g. sigmoid-scaled/leaky) and large-signal loss on field magnitude out vs in.
3. D1+: ship s~100 softcap as inference default for contact-C ckpts; combine with D3b/D5 if they leave residual runaway.
4. Shift-consistency loss with random sub-cell shifts (D6 supports); needs same-schedule control, count dE rows as failing.
5. State assimilation (D4): evaluate noisy pos/vel observation for close pairs; only if a deployment with observations is in scope.
6. Fine-tune dE regression: every fine-tune fails nr100 dE@1000 / bh300 dE; investigate that (energy soft loss, lower lr, freeze base) as its own item before more fine-tune variants.
7. Deferred batch from the original list (edge guard band, Galilean boost, restitution/energy loss, distributional loss, failure restarts, anti-aliasing, denoiser).

## Rejected / already tried (do not repeat)
Closed-loop relabel curriculum (negative), thermostat/L-rescale, parareal, Kalman, far quadrupole, GroupNorm, circular padding, output low-pass/spectral norm.
- D5 (k_end 50, lr 1e-4 staged, 2026-10-13): longer horizon does not fix Luxis C40 runaway; lower-lr control removes it, all fine-tunes still fail dE rows.
- E1 (2026-10-13, branch fix-finetune-dE, write-up docs/debugging/error-correction-ft-dE.md there): any U-Net weight update takes nr100 dE@1000 .02 -> .2-.5 in ~25 steps (lr 1e-4 and 1e-5, seeds 4738/4739). Failed on Exp E: soft dE loss w1 (.045, err@20 +33%), lr 1e-5 warmup (.275), EMA .999 (.313), interpolating toward fine-tuned weights (25% already .063). Only heads-only (kmlp/ppmlp, U-Net frozen) passes nr100 dE (.0218 vs .0170) but no accuracy gain and fails nr100 dL, bh300 dL, bh300 dE@100. Control fine-tune's harness gain not real on 48 fresh scenes. DECISION 2026-10-13: heads-only rejected (worse err, no benefit); do not ship or revisit. Option C with U-Net frozen: err@20 7.92, tunnel@20 59%, leaves grid by tick 30. Harness dE rows are ill-defined for C (learned gravity vector, contact head). Open: C geometry branch w/ frozen E U-Net, anchor loss on no-geometry force, soft dE w .1-.3 on C.
- D2b/E3 (2026-10-13, branch worktree-agent-a226cff35f933b29f, doc error-correction-D2b-contraction.md there; jobs 3547-3577): sigmoid forget gate + large-signal growth penalty on recurrent field. Gate-only on frozen C: no effect. Full fine-tune with penalty (linear hinge; log hinge; gmax .5): 3/3 diverge (1804 rollbacks, D2b2 err@20 ~100) vs stable same-code control. Staged recipe (D5ctrl/D2bctrl) removes natural runaway but runs away 88-100% under x100 field injection, so it is not contractive. All fine-tunes fail harness dE rows. Untried: penalty without gate, penalty weight schedule, grad-norm guard.
- E2 softcap (2026-10-13, branch worktree-agent-a51e0b89567ace982, doc error-correction-softcap-ship.md there; not merged): tanh softcap s=100 on fed-back field as inference default for contact-C: contact runaway .50/.67 -> 0, err@20 +5-12%; D5ctrl+s100 clears expB gravity runaway (cluster 17-25%); dE rows still fail. Works as a mitigation, not a fix.

- Root-cause decomposition (2026-10-14, jobs 3601-3607, write-up long-rollout-root-cause.md, scripts scatter_rootcause*.py, uncommitted): E's expB trajectory error is reproduced by exact Verlet + iid 8% relative force noise (err@20/50/100 .0256/.168/1.54 vs E .0254/.169/1.47), so force amplitude of nn<1 bodies is the proximate cause; the same architecture trained on B data has 0.94% force error (err@100 .42) and zero-parameter untruncated P3M pp=8 has 0.6% (err@100 .41) vs E 6.1% (1.47): floor is learned, not grid-intrinsic (pp=2 analytic 12.5%). Integrator floor (Verlet-1 vs 4-substep truth) binds in clusters (E already at it for ticks >= 50) and for expB below ~1% force error. 4-phase shift averaging: E err@20 -39%, @100 -30% (inference-only, 4x cost). E's first Adam step (lr 5e-4) took expB force 0.94% -> 5.7%, never recovered in 60k it.
- pairfull (2026-10-14, branch rootcause-pairfull, job 3604, one seed): analytic P3M base + pp=8 + untruncated neighbours, 4000 it vs same-schedule ctl: ab err@20 .0138 vs .0286 (-52%), cl .0345 vs .0579, harness PASS (ctl FAIL bh300 dE/dL); but untrained base is better still (ab err@20 .0050, force 0.68% vs trained 4.8%): training degrades the base (median body err x2.4 while held-out MSE -12%, p99 -16%; top 1% bodies carry 36-91% of loss). nr100 dE@1000 .057 vs E .017. Verdict: positive vs ctl, not shipped; next single change = robust (Euclid-norm) rollout loss on the same base.

- Substep theory oracle (2026-10-14, jobs 3614/3615 CPU, exact force only, write-up substep-theory.md, script scatter_substep_theory.py, uncommitted): Verlet-1 floor = ~0.7% (expB) / ~2% (cluster) iid force noise; integrator is 0.8% of E's squared err@20 on expB (no gain possible), 27% cluster. Per-body adaptive substeps and frozen-far/one-sided-kick variants are WORSE than Verlet-1; Yoshida4 and jerk-extrapolated drift worse too. Symmetric RESPA (grid once/tick half-kicked, pair term within rc on M substeps, all bodies) works: err@20 expB/cluster rc4 M4 .00027/.0028, rc8 M4 .0001/.0014 vs Verlet-1 .0023/.035 and KDK2 .00045/.0052. Next: inference sub2/sub4 on pairfull0 (kill if cluster @20 improves <20%).

- Euclid rollout loss at full Exp E recipe (2026-10-15, branch loss-euclid-fullscale, jobs 3642/3648, log in experiment-log.md): 60769 it warm-from-B, seed 4738, single change vs same-schedule squared control. ab err@20 .0217 -> .0063 (-71%), err@100 1.38 -> .715, cl err@20 -41% (err@100 -8%): accuracy win survives at full scale. Conservation fails: nr100 dE@1000 median of 3 .198 vs control .105 (frozen E .027; gate .05), bh300 dL@100 1.15 vs .068. Verdict KILL on pre-registered energy criterion; no replicate. Open: Euclid + anything that restores nr100/bh300 conservation (all prior energy add-ons failed, see E1/energy-sim); the squared control itself also fails nr100 dE.

## Batch 2 dispatched (2026-10-13)
Three agents, each in its own worktree, sharing the one-GPU-job-at-a-time Polaris queue:
- E1: fix fine-tune dE regression (follow-up #6): energy soft loss / lower lr / frozen base / EMA, vs frozen C + same-schedule control
- E2: D1 softcap s~100 as inference default for contact-C + staged low-lr recipe (follow-up #3)
- E3: D2b contraction retry, nonzero-grad gate + large-signal loss; control = staged recipe without contraction (follow-up #2)
Saved for later if these fail: follow-up #4 shift-consistency loss, #5 state assimilation (D4), #7 deferred batch (edge guard band, Galilean boost, restitution/energy loss, distributional loss, failure restarts, anti-aliasing, denoiser, PDE-Refiner, time-reversal), D3b rerun with better metrics.

## Decision 3 (2026-10-13)
Harness energy rows: option 3. Keep geometry-free dE rows (nr100@1000, bh300@1000, geometry/gravity vector/contact head zeroed) as the gate every contact model must pass; add contact-specific energy rows (wall scenes + gravity vs same-physics truth, `scripts/contact_truth.py`) reported alongside, counted only after the gate passes. Not yet implemented. Test-suite review agent dispatched same day.
- B-frozen / D2 option B (2026-10-13, branch worktree-agent-a61e7dc4af33d7032, doc error-correction-B-frozen.md there): SDF geometry branch (+gvec, +optional pair head) on fully frozen Exp E. Stable: tunnel 0%, runaway 0, off-grid 0, err@20 4.1-5.1, err@100 median 3.5-5.8 incl. held-out disc/post_poly (vs C ~1.4e3, A2 ~4e4, frozen-C ~175). But over-damped like B2: KE ratio@100 ~3e-4, restitution .4-.5 at v=15; video review: Bf1 stacks balls, Bf2 static column. Harness nr100 dE .0177 -> .0235 (dL .028 -> .075): near but not exactly E. Over-damping not fixed by freezing (supports multi-step-MSE cause).

- Recurrent-state ablation on Exp E (2026-10-14, branch worktree-agent-a851652ea9f780ebf; jobs 3592 norec, 3598 rec2 control resumed from it20000 after 3593 diverged, eval job 3600): norec = ms_kp_pot_v_g128_norec (no fed-back field), same B->E recipe. Pre-launch prediction (norec +20-50% err@20, larger @100, stability >=) MISSED. err@20/@100 expE mean: norec ~.037/2.81 vs rec2 ~.039/2.84 (<5%, within noise); expB err@100 1.33 vs 1.45; bh300 err@100 4.90 vs 5.03. Conservation (24 held-out scenes, 1000 ticks, median |dE|/scale): nr100@1000 norec .127 vs rec2 .040; nr10@1000 .049 vs .038; bh100@1000 .296 vs .182; bh300@1000 .374 vs .867 (rec2 ejects 50%, virial 1.96). 1e4 ticks: bh300 rec2 collapses (ejected .99, virial 485), norec ejected .46 but virial 1.2; nr100 norec dE 1.06 vs rec2 .39. Truth t4 dE is 3e-5..1e-3, so model is 100-1000x worse everywhere. Verdict: recurrent field does not set long-horizon accuracy; it trades nr (helps energy) against bh300 (hurts stability). Not a fix; no ship. Single run per arm, ~5% diffs are noise.
- Rollout loss function (2026-10-14, branch loss-tests, jobs 3612/3618, write-up loss-function-tests.md, scripts scatter_losstest*.py, uncommitted): position term squared -> per-body Euclid norm, pp=2 ms_kp_pot_v_g128, 4000 it, scratch and warm-from-B setups, seed 4738 + replicate 4739. POSITIVE: ab err@20 .0262 -> .0116 (scratch), .0245 -> .0063 (warm), err@100 1.48 -> 1.03 / .74, replicated; warm median step-20 error back to B level (.0049 vs .0043; sq-loss .0218); cl err@20 -32 to -41%. Huber (delta .0029 = B median error) = ctl, inconclusive: position term ~350x down-weighted vs velocity term (confound). Harness: all arms FAIL dE rows; Euclid worst on nr100 dE@1000 (S 2.12 vs sq .125; W 1.77 vs .39), single run. Next: Euclid loss on the pairfull base / full-scale E recipe, gate on dE rows.
- Substep router (2026-10-14, branch substep-router, jobs 3613/3621, write-up substep-router.md there, script scripts/substep_router.py, uncommitted; exact force only): per-body substep promotion (block KDK levels 1/2/4; oracle, tiny MLP, Aarseth, omega, nn<1, closure, pair-flagged RESPA, per-tick global M). One-step error is easy to classify (oracle top 10% = 70-94% of squared one-step error; MLP precision/recall .93-.98 vs oracle) but does not convert to rollout gain: expB oracle router cost 1.42 err@20 .00235 vs U1 .00232, U2 (cost 2) .00045; nothing at cost <= 2.4 within 1.5x of U2 in any family. Promoted fraction at fixed tol rises with density (.28 expB -> .88 bh3000), so savings vanish where substeps matter; block kicks break |P| (1e-4..1e-2). Per-scene scheme classifier: x1.04 over best global at err*cost. Verdict: reject per-body/per-scene routers; use uniform symmetric RESPA (rc>=4, 4 substeps on all near pairs; expB err@20 .00019, cluster .0019) or uniform KDK2.

- Loss term weighting (2026-10-14, branch loss-weights, jobs 3622/3626, write-up loss-term-weighting.md, uncommitted), vs same-schedule Euclid ctl, warm (and scratch): A Euclid velocity term: warm ab err@20 .0065 -> .0055 (-15%; rep -18%), err@100 .746 -> .663 (rep -15%), BETTER warm only; scratch WORSE (+19% err@20); bh300 dE@1000 warm 3.3 vs 1.76. B step weights ~ Exp B per-step median error (w_1 .09, w_20 2.7): noise (+5% / +3%). C unit-slope Huber (delta .00286): noise warm (0%), worse scratch (+15% err@20, median +33%); earlier Huber failure not just scale mismatch. All arms FAIL harness dE rows; nr100 dE@1000 ranges .075-1.76 with no pattern (noise). Untried: w ~ 1/e_s (early-step upweight), A on pairfull/E-scale.

- Force-anchored aux loss (2026-10-14, branch loss-force, jobs 3624 seed 4738, 3628 replication seed 4739, write-up loss-force-anchor.md, uncommitted), vs Euclid ctl: warm +exact (lambda .32 on nn<1 bodies) expB force 1.41% -> 0.85%, ab err@20 .0063 -> .0049 (-22%), err@100 .721 -> .591 (-18%); warm +anchor (frozen B) similar (.0053 / .579) but cl force worse (7.7% vs 6.2%), bh300 dE@1000 3.8. cl err@20 flat (.0346 vs .0363). Scratch +exact WORSE (ab err@20 .0125 vs .0090) despite force 1.18 vs 1.35%. Harness: all arms FAIL bh300 dE/dL@100 as every arm; nr100 dE@1000 not better (1.9 vs 1.35). Job 3628 (agent-reported, per-seed table not in doc; scratch_exact_s4739): position error down 54-81% but bh300 dE@100 .048 -> .365, dL@100 .30 -> 3.6, i.e. fails regression. Verdict: helps warm ab only; does not fix conservation or clusters. Untried: aux on cluster-scale states, lower lambda.

## Write-up locations

Docs cherry-picked from unmerged branches (branch in parentheses):
- error-correction-softcap-ship.md (worktree-agent-a51e0b89567ace982)
- error-correction-D2b-contraction.md (worktree-agent-a226cff35f933b29f)
- error-correction-ft-dE.md (fix-finetune-dE)
- error-correction-B-frozen.md (worktree-agent-a61e7dc4af33d7032)
- test-audit-2026-10-13.md (test-audit-2026-10-13)
- scatter-contact-c.md, scatter-contact-c-table.md, scatter-correction-research.md (worktree-agent-a51e0b89567ace982; identical on several other branches)
- contact-grid-review-prompt.md (fix-finetune-dE)
- Euclid-loss energy-fix ladder (2026-10-19, branch euclid-energy-ladder, job 3752, log in experiment-log.md): from frozen E, Euclid loss, 10k it lr 1e-4 k 20, seeds 4738/4739. Run 1 heads-only: ab err@20 .0233 -> .0216 (-7.5%, gate <= .0076 FAIL), nr100 dE@1000 .0055/.0174 PASS. Run 2 full + L2-SP lam 5: same accuracy, nr100 dE .16/.19 FAIL (dE hypersensitive to ~1e-5 rms weight shifts). Run 3 = Run 2 + nr100/bh300 replay: nr100 dE .18/.16 FAIL, bh300 dL@1000 improved. No arm passes; all arms reach the same accuracy, so the short schedule is the limit. Untried: heads-only at the full 60k-it lr 5e-4 schedule; larger lam.
