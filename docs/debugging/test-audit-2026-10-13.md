# Test and gate audit, 2026-10-13

Branch `test-audit-2026-10-13` (from master f0e6238). Scope: every pytest test, script-style test, harness gate row and web test. Judged against: scatter-field E (E_ms_kp_pot_v_g128), contact B/B2/C, token-per-ball, gravity/N-body, web viz. CFD (ContactForceDynamics) = stale-by-scope, listed only. Units: err@k = mean per-body position error in sim units.

Method: ran what is CPU-light locally (pytest tests/, JS tests under node, headless-chromium web tests, CPU drivers for GPU-labelled script tests), read code for the rest. GPU-only claims are marked UNTESTED. Experiment log: "Test-suite audit experiments" in error-correction-plan.md (A1-A7).

Caveat on scope: option C, `--contact/--contact-free` harness flags, `ft_de_test.py`, `bfrozen_test.py`, `scatter_softcap_test.py`, `scatter_d2b_verify.py` and the modified `scatter_contact_test.py` live on unmerged branches (fix-finetune-dE, a51e0b89, a226cff3, B-frozen a61e7dc). On master `cs.build` only knows E/augur1/B/A (scripts/scatter_contact_scenes.py:152-163). Those files were NOT audited beyond listing; the harness diff on fix-finetune-dE was read.

## Summary: which signals to trust

Trust: tests/ (230 pass, 26 s CPU); JS ports vs torch (gravity/scatter/token verify, 1-step diff 2e-7 observed); scatter web browser tests; harness err@10/err@20 rows for expB and bh300, bh300 err@5; harness timing rows on identical hardware and opt stack; nonfinite row; `scatter_contact_test` geometry parts; `scatter_opt_test` resolve check.

Do not trust (reasons in table): harness two_body err@5 (30 % tol vs per-seed spread 30 %); nr100@1000 dE and bh300 dE@1000 / dE@100 / dL rows (4-scene medians, spread wider than tolerance; bh300 @1000 baseline already 45-64 % energy error); every dE/dL/bound/virial row for contact models (gravity vector and contact head in the force); peak_mem_mb (no valid baseline, graph pools retained); bh300 vmax/c<1 and frac|v|>=c (tautological); bound/virial gap rows for two_body and expB (atol 20-60x the baseline value); anything in tests that only prints (scatter_opt_test parity, physics.test.mjs before this branch).

## Table

Verdicts: valid | stale | broken | buggy | bad-signal | redundant. Counts at the end.

### pytest tests/ (230 tests, all pass)

| # | test/gate | verdict | evidence | action |
|---|---|---|---|---|
| 1 | test_token_* (curriculum, dataset, detect, free, gate, graph, losses, match, model, net, rasterize, refine, split, track_query, train) 158 tests | valid | pass; every test has >=1 assert (AST scan); mocks in test_token_train.py:56,75,141 gate arguments, not the result | none |
| 2 | test_tiled_sim.py (2) | valid | zeroing dynamics weights moves the 12-tick rollout by 43.9 units vs tol 1e-3 (A4), so tiled-vs-global does exercise forces | none |
| 3 | test_gravity_tiled.py, test_central_force.py (3) | valid | random weights std .3, atol 1e-4, real force | none |
| 4 | test_contact_truth.py (2) | valid | GPU truth vs augur.py CPU reference, 1e-4; the reference Decision 3 contact rows will use | none |
| 5 | test_bounce_contact.py (12), test_obstacles.py (6), test_dataset.py (7), test_eval_free_rollout.py (4), test_losses.py (11) | valid | model/dataset.py, losses.py are imported by token_train/token_losses and ~15 scripts | none |
| 6 | test_net.py (8), test_evaluate.py (3), test_train.py (3 rollout_loss tests) | stale | Stage-2 flow-warp line (model/net.py, evaluate.py, train.rollout_loss), superseded by token-per-ball per memory; evaluate.py referenced only by scripts/compare_rollout.py. test_evaluate asserts `loss >= 0` and not-NaN only (tests/test_evaluate.py:12,26) | kept (code still in repo); delete together with model/net.py, evaluate.py if the line is retired. User decision |
| 7 | test_contact_force.py (8) | stale | CFD, stale-by-scope | none (listed) |
| 8 | test_train.py::test_sampling_probability | valid | shared with token_train | none |
| 9 | bare `pytest` at repo root | broken | A1: 2 collection errors, scripts/scatter_contact_test.py:141 `main()` at import, scripts/test_web_ui.py module-level chromium launch; with `-q` argv became the ckpt path | FIXED (conftest collect_ignore_glob), commit |

No pytest exists for the scatter-field model (ScatterField, ContactScatterField, harness compare logic). Coverage gap: the compare() logic itself (leaked-variable bug below went unnoticed).

### Harness `scripts/scatter_regress.py` (+ polaris_scatter_regress.sh)

Baseline: results/baseline_v2_regress.json (E, job 3348), 4 seeds x 48 scenes (two_body, expB), 1 bh300 scene per seed, conservation 4 scenes per regime (scatter_conservation.REGIMES per=1, SEEDS has 4 seeds).

| # | row | verdict | evidence | action |
|---|---|---|---|---|
| 10 | two_body err@5 (tol 30 %, atol 1e-5) | buggy | baseline per-seed 0.0007/0.0015/0.0007/0.0013 (rel std .301, results/scatter_baseline_v2.json). Value 1.1e-3 is 20x the 1e-5 perturbation floor (chaos_floors.json pert_1e-05 5e-5), so not floor-limited; spread is scene-to-scene, but the relevant noise for a retrained candidate is fine-tune variance 10-15 % (memory feedback_select_on_heldout_seeds). A 30 % tol on a 1e-3 quantity false-fails equal models (GPU reproduction UNTESTED) | propose P1 |
| 11 | err@10, err@20 two_body; err@5/10/20 expB, bh300 (tol 15 %) | valid | expB rel std 2-4 %, bh300 7-11 %; floors 100x below model (chaos_floors.json) | none |
| 12 | bh300 err@50, expB err@50/100: not gated | buggy (missing gate) | policy docs/debugging/scatter-baseline.md:63, scatter-chaos-floors.md:5: BH headline = err@50 (floor .006-.015 vs model .91). compare() gates only 5/10/20 (scatter_regress.py:391), err@50/100 printed as info (:432) | propose P4 |
| 13 | bound_frac gap@100, virial gap@100, two_body and expB (tol 15 %, atol .02 / .05) | buggy | scatter_regress.py:38,399. Baseline gaps: expB bound .0009, virial .0008; two_body bound .0052, virial .034. atol .02/.05 means expB must worsen 20x/60x and two_body virial 2.6x before FAIL | propose P8 |
| 14 | bound/virial gap bh300 @100 | valid-weak | base bound .0067, virial .048 ~ atol .05; 4 scenes | propose P8 |
| 15 | bh300 dE_rel@100, dL_rel@100 (tol 25 %) | bad-signal | 4 scenes; per-seed dE .0748/.0229/.0164/.0493 (rel std .55 > tol .25). ft-dE doc: bh300 dE@100 harness FAIL for heads-only was "4-scene noise: 12-scene probe .0587 vs base .0592" (docs on branch fix-finetune-dE) | propose P2 |
| 16 | bh300 dP@100 (atol 1e-3) | bad-signal | baseline 4e-5, atol 25x baseline | propose P8 |
| 17 | bh300 vmax/c < 1 (mean over seeds) and cons frac\|v\|>=c | bad-signal | harness applies `speed(mom,c)=p/sqrt(1+p^2/c^2)` itself (scatter_regress.py:49,223, scatter_conservation.py:roll_model), so |v|<c holds for any model; the row cannot fail. The mean over seeds (not max) would also hide a single-seed violation | propose P5 |
| 18 | cons nr100 dE@1000 (median of 4 scenes, tol 25 %, atol 1e-3) | buggy | base per-scene dE spread p16-p84 = -0.0002..0.0363 (results/conservation/summary.json, B=4); 24-scene probe .0206 [.011,.034] vs harness .0170 (ft-dE doc). Gate tolerance 25 % is inside both the 4-scene vs 24-scene gap (21 %) and the p16-p84 band (+-60 %). Known failures are 10-25x (ctrl .22-.48), so the row separates only gross failures | propose P2 |
| 19 | cons bh300 dE@1000 | bad-signal | base 4-scene .481-.45, 12-scene .64 [.14,1.0] (ft-dE doc); model loses ~45 % of energy in 1000 ticks with 29 % bodies off grid (summary.json bh300 m: out .29); truth 4 substeps -0.0004, 1 substep +0.23, so the quantity is saturated and integrator-sensitive; "no worse than 1.25x baseline" locks in a broken reference | propose P3 |
| 20 | cons dL, dP, bound_gap, vir_gap, outgrid @1000 | buggy | same 4-scene medians; dP atol 1e-3 vs ~1e-6 real values (probe: P median 7e-7) | propose P2, P8 |
| 21 | cons nonfinite | valid | cannot be satisfied by a degenerate model | none |
| 22 | all dE/dL/bound/virial/err rows applied to contact models (C) | bad-signal | ft-dE doc: C fails 27 rows incl err@5 .64 vs .001 because gvec (+x gravity) and contact head are in the force; with gvec/cmlp zeroed C still fails 27 (nr100 dE 1.9e4), Cfz 21 (gscale drifted to 1.28). Rows measure a different physics. Harness `--contact`/`--contact-free` exist only on fix-finetune-dE (10548f4, c42a282) | Decision 3 (below) |
| 23 | tick_ms (tol 5 %) | valid-conditional | 5 reps spread <0.1 % (baseline tick_ms_all). Launch-bound: two_body (B48,N2) 4.73 ms vs bh300 (B1,N300) 2.48 ms vs expB_batch1_n100 2.46 ms, i.e. insensitive to N. compare() never checks meta.opt/gpu/torch (scatter_regress.py:422); baseline_v2 meta opt '' (eager) vs default graph stack is a cross-config comparison that passes only because graph is faster | propose P7 |
| 24 | peak_mem_mb (tol 10 %) | broken | (a) committed baselines have no mem_def: `compare` prints "memory gate SKIPPED: baseline None" (A6), so the gate is dead. (b) mechanism, from code: `_graphed` caches (graph, static clones, outputs) per input shape in `model._graphs` for the model's lifetime (scatter_field.py:78,421-438); `reset_peak_memory_stats` in timed() (scatter_regress.py:93) resets to current allocated, which includes pools retained from earlier regimes, so graphed peak depends on regime order and on how many distinct n_max captured. Eager baseline 87-889 MB per regime is per-run. Numeric reproduction UNTESTED | propose P6 |
| 25 | compare(): leaked `reg` in bh300 dE/dL/dP rows | buggy (latent) | scatter_regress.py:404-406 used `reg` from the previous loop (== "bh300" only because it is last in REGIMES); no output change today | FIXED, commit |
| 26 | floors table, ood_battery, info line | valid | not gated; floors definitions match chaos_floors.json | none |
| 27 | CFHarness / `--variant cf` | valid | comparator for CF-gap docs | none |

### Script tests, gates and probes

| # | item | verdict | evidence | action |
|---|---|---|---|---|
| 28 | scripts/scatter_contact_test.py truth parity, sdf, gather, knn | valid | CPU run 13/13 PASS (A5) | guard main (done) |
| 29 | scatter_contact_test identity/grad | valid-untested + stale parts | needs ckpt (UNTESTED). Covers E/B/A only; A (kernel) diverges past k=20 and is unusable (scatter-contact.md); B2/C not covered; `check()` records instead of asserting, so pytest would pass vacuously (it is no longer collected) | add C/B2 identity when C merges |
| 30 | scripts/scatter_fix_tests.py far_multipole | valid | CPU PASS; mono rel err .128 at R=20 vs assert <.2 (loose 1.5x); quad (`farquad`) is on the rejected list (far quadrupole) but the code is still opt-in | none |
| 31 | scatter_fix_tests flags_off_identity | valid-weak | with no ckpt random-init weights (near-null force); substeps=2 diff 9.5e-7 not asserted; d_far==0 trivial for scene inside .75 of box | run with --ckpt on GPU |
| 32 | scripts/scatter_opt_test.py check_resolve | valid | CPU PASS | none |
| 33 | scatter_opt_test inference/train parity, PPCAP, CELLGRAPH | buggy | prints max diffs only, no assertion (scatter_opt_test.py:57-121); NOISE eager-vs-eager runs the same deterministic model twice | propose P9 |
| 34 | scripts/test_graph_step.py | redundant | no assert; Exp A ms_kp_pot; parity in scatter_opt_test --train-opts | DELETED, commit |
| 35 | scripts/test_web_ui.py | broken->fixed | A2: 2 fails + crash at s-zoom | FIXED, commit; 22/22 pass |
| 36 | tile_box_test.py, tile_speed_test.py, gravity_tiling_test.py | valid-as-experiment | sweeps with no pass/fail; tile_speed/tile_box target cons_pure (token conservative) | not gates; rename optional |
| 37 | eval_free_rollout, eval_trivial_baselines | valid | report-only | none |
| 38 | eval_step1_baseline.py | stale | Stage-2 flow-warp diagnostic | none |
| 39 | eval_contact_generalization.py, eval_energy_fix.py | stale | CFD / token energy-fix history, stale-by-scope | none |

### Web tests

| # | item | verdict | evidence | action |
|---|---|---|---|---|
| 40 | gravity tests/trace.test.mjs, truth.test.mjs | valid | run: trace diff 1.8e-15, truth attracts OK | none |
| 41 | gravity verify.mjs, token verify.mjs | broken-by-missing-artifact | read tests/vectors.json, which is gitignored and absent in a fresh checkout (ENOENT); regenerate with scripts/export_web_testvectors*.py (needs torch ckpt, not run, UNTESTED). Gate 1e-4 on 1-step vs observed scatter 2e-7 | note in web READMEs |
| 42 | token physics.test.mjs | buggy->fixed | exited 0 always; n20_40 f200 = 18.7 | gate f1<=1e-9, f20<=1e-5 added (new gate, not a threshold change) |
| 43 | scatter verify.mjs | valid-loose | pass; 1-step worst 2.3e-7 vs gate 1e-4 (400x loose); only t=0 gated | propose tighten to 1e-6 |
| 44 | scatter truth.mjs | valid-loose | pass; drift 4.8e-4 vs gate 5 %; parity to python @10/@100 1e-13 | none |
| 45 | scatter frames.mjs, precomputed.mjs, tests/*.py (export, seed, model), walkthrough verify.mjs, interact.py, zoom.py | valid | all pass locally (A5-style run); precomputed bit-exact regenerate N=10 only (other N checked for tick 0) | none |
| 46 | scatter-walkthrough ui.py | valid-weak | screenshot driver; only asserts no console errors | none |
| 47 | walkthrough verify "accel = gather + pair - mean", "gather matches independent CIC sample" | valid-weak | identity against own formula; both diff 0.0 | none |

## Counts (rows above; a row may cover several tests)

valid 16 (rows 1-5, 8, 11, 21, 26-28, 32, 37, 40, 44, 45); valid with caveats 9 (14, 23, 29, 30, 31, 36, 43, 46, 47); stale 4 (6, 7, 38, 39); broken 4 (9, 24, 35, 41); buggy 8 (10, 12, 13, 18, 20, 25, 33, 42); bad-signal 5 (15, 16, 17, 19, 22); redundant 1 (34). Total 47 rows. Fixed on this branch: 9, 25, 28 (import guard), 34 (deleted), 35, 42.

## Proposed corrected gate definitions (NOT applied; thresholds are the user's call)

P1 two_body err@5: info only, gate err@10/err@20. If kept: fail only if paired scene-level ratio > 1.15 and (cand-base)/pooled SEM > 3 (err_sem is already stored per regime).
P2 conservation rows: 24 scenes nr100, 12 bh300 (ft_de_probe.py already does), fail if cand median > max(2 x base median, base p84). Rationale: known fine-tune failures are 10-25x (.02 -> .22-.48); the 2x rule tolerates the .011-.034 base band. Under it pen_e1 (.0449, 2.2x) still fails, heads-only (.0218) passes.
P3 bh300 dE@1000: info only (saturated at .45-.64, truth integrator-sensitive). Keep bh300 dE@100 on 12 scenes.
P4 add err@50 for bh300 (floor .006-.015, base .91) and expB (floor .0006-.0009); err@100 only for expB (floor .013-.019 vs 1.42) as policy already states.
P5 delete bh300 vmax/c<1 and frac|v|>=c from the harness gate, or compute them on model-native velocity (token model only).
P6 memory: clear `model._graphs` and run each regime in a fresh process (or report graph pool separately); store mem_def in baselines; regenerate baseline_v2 with mem_def.
P7 compare(): require equal meta.opt, gpu, torch for timing/memory rows, else skip with a message.
P8 bound/virial/dP atols: scale to baseline (expB bound .005, virial .005; bh300 dP 1e-4) or gate on a 3 x noise multiple measured by chaos_floors.
P9 scatter_opt_test: assert inference max|dpos|@20 < 1e-3 (graph/cell flags) and train grad rel diff < 1e-3 after measuring noise on GPU (UNTESTED).

## Decision 3: what the harness rows need (not implemented here)

Geometry-free gate (kept): `scatter_regress.py run --contact --contact-free` (fix-finetune-dE) zeroes gvec and the contact head but leaves gscale (Cfz gscale drifted to 1.28 and then fails err@5 .12). Needs: pin gscale = 1 in `cs.free_space`; merge `--contact/--contact-model` and the `model.dt = args.dt` fix from c42a282 to master; apply P2/P3 so the gate is not noise.
New contact rows (separate, counted only after the gate passes), using scripts/contact_truth.py and scatter_contact_scenes truth (k=400, gravity 9 toward +x, dt .15, 8 substeps): wall-scene energy and gravity vs same-physics truth; restitution probe (B2: .49 at v=12 vs truth 1.0; low-speed probe returned None, "may be the probe failing"; verify first); KE ratio vs truth at tick 100 (B .003-.009); tunnel fraction; off-grid fraction and err@100 median (D3b lesson: mean err@100 hides ejected balls); recurrent-runaway rate |a|>200; err@20 on c_id/c_nonconvex, 72 scenes not 24.

## Decisions needed (separate)

1. Adopt P1-P9 (each changes pass/fail behaviour).
2. Retire the Stage-2 line tests together with model/net.py, evaluate.py (row 6).
3. Regenerate baseline_v2 with mem_def, 24/12-scene conservation baselines.
4. Merge the option-C harness flags to master.
