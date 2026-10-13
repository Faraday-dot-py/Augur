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

## Batch 2 dispatched (2026-10-13)
Three agents, each in its own worktree, sharing the one-GPU-job-at-a-time Polaris queue:
- E1: fix fine-tune dE regression (follow-up #6): energy soft loss / lower lr / frozen base / EMA, vs frozen C + same-schedule control
- E2: D1 softcap s~100 as inference default for contact-C + staged low-lr recipe (follow-up #3)
- E3: D2b contraction retry, nonzero-grad gate + large-signal loss; control = staged recipe without contraction (follow-up #2)
Saved for later if these fail: follow-up #4 shift-consistency loss, #5 state assimilation (D4), #7 deferred batch (edge guard band, Galilean boost, restitution/energy loss, distributional loss, failure restarts, anti-aliasing, denoiser, PDE-Refiner, time-reversal), D3b rerun with better metrics.

## Decision 3 (2026-10-13)
Harness energy rows: option 3. Keep geometry-free dE rows (nr100@1000, bh300@1000, geometry/gravity vector/contact head zeroed) as the gate every contact model must pass; add contact-specific energy rows (wall scenes + gravity vs same-physics truth, `scripts/contact_truth.py`) reported alongside, counted only after the gate passes. Not yet implemented. Test-suite review agent dispatched same day.

## Test-suite audit experiments (2026-10-13, branch test-audit-2026-10-13, full write-up docs/debugging/test-audit-2026-10-13.md)
- A1 bare `pytest` at repo root: 2 collection errors (scripts/scatter_contact_test.py ran main() at import and opened ckpt '--collect-only'; scripts/test_web_ui.py launched chromium at import). `pytest tests` alone: 230 passed, 26 s CPU. Fixed by conftest collect_ignore_glob; root collection now 230 tests, 0 errors.
- A2 scripts/test_web_ui.py against current web/: FAILED 2 checks (landing has 4 cards not 3; scatter header now has a back link) then CRASHED at the zoom block (s-zoom/s-centre/b-fit removed when the 3D model view became the scatter page). Updated; 22/22 pass.
- A3 web/token/tests/physics.test.mjs: exit 0 regardless; n20_40 f200 diff 18.7 (chaos), f1 0, f20 2.7e-7. Added gate on f1/f20 only.
- A4 mutation check of tests/test_tiled_sim.py: zeroing the dynamics weights changes the 12-tick global rollout by 43.9 pos units vs the test tolerance 1e-3, so the tiled-vs-global test does exercise pair/wall forces (not a weak test).
- A5 CPU runs of GPU-labelled script tests: scatter_opt_test.check_resolve OK; scatter_fix_tests far_multipole + flags_off_identity OK (random weights, no ckpt); scatter_contact_test truth/sdf/gather/knn 13/13 PASS; identity/grad need a ckpt (untested).
- A6 harness compare of baseline_v2_regress.json against itself: OVERALL PASS but "memory gate SKIPPED: baseline None" (committed baseline predates mem_def; the memory gate has no valid baseline).
- A7 NOT reproduced on GPU (Polaris queue left to the B-frozen agent): peak-memory order dependence under CUDA graphs, two_body err@5 false-fail on a retrained candidate. Evidence for both is from code reading and existing JSONs; marked untested in the audit.

## Audit decisions executed (2026-10-13, all four delegated to the audit agent's recommendations)
- Item 2 (retire Stage-2 tests, net.py, evaluate.py): NOT retired. Live importers of model.net / model.evaluate remain: scripts/render_diagnostic_grid.py, render_rollout_video.py, compare_rollout.py (also model.evaluate), eval_step1_baseline.py, investigate_gridding_artifact.py, investigate_long_horizon_lattice.py, plus tests/test_train.py (rollout_loss on model.train). render_diagnostic_grid.py and render_rollout_video.py are the documented flow-warp diagnostic renderers (experiment-log.md, drift briefs). Retiring needs deleting those six scripts and their doc references; not done without the user choosing that.
- Item 4: merged worktree-agent-a61e7dc4af33d7032 (B-frozen branch; contains --contact/--contact-model/--contact-free from 10548f4 and model.dt = args.dt from c42a282) with --no-ff; no conflicts. gscale pinned to 1 in cs.free_space (61b66a3).
- Item 1: P1-P9 applied, one commit each (see git log). P1 err@5 two_body info only. P2 24 nr100 / 12 bh300 conservation scenes, per-scene arrays stored, fail if cand median > max(2 x base median, base p84), bound/virial gaps per scene. P3 bh300 dE@1000 info only, bh300 dE/dL/dP@100 gated on the 12 scenes (old 4-scene rows removed). P4 err@50 gated for bh300/expB, err@100 for expB. P5 vmax/c row and frac|v|>=c gate removed. P6 harness drops cached CUDA graphs before each timed regime, mem_def string updated. P7 timing/memory rows skipped unless opt/gpu/torch match. P8 bound/virial atols .005/.005. P9 scatter_opt_test asserts (infer max|dpos|@20 < 1e-3, train grad rel diff < 1e-3), values are the audit's proposals pending GPU measurement.
- tests/test_scatter_regress_compare.py covers compare() (9 tests, CPU).
- Legacy baseline kept at results/baseline_v2_regress_legacy.json (4-scene cons, no mem_def, opt '' eager).
