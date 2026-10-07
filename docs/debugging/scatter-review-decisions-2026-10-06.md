# Scatter-hardening review: user decisions (2026-10-06)

Walkthrough of the 10 agents (baseline, chaos, optimization, scaling, conservation, OOD, scenes, scatter-vs-CF gap, tick-8109, causal field). Decisions are recorded here and NOT applied until all 10 are answered.

## 1. Baseline harness
Proposed: (a) pooled-seed or ~30% gate on two_body err@5; (b) replace peak-memory metric for graphed runs; (c) add bound fraction / virial / |v|<c rows to the gate; (d) state "Exp E only" in docs.
Decision: ACCEPT all four (a-d). Memory metric = option 1 (reset_peak_memory_stats before warm-up, report absolute max_memory_allocated, same definition for eager and graphed); log max_memory_reserved alongside as info only. Not yet applied.

## 2. Chaos floor
Proposed: (1) stop quoting BH err@100, headline BH at err@50, err@100 only for Exp B; (2) add reference-noise floors row to harness (BH err@50 < .015, Exp B err@100 < .01 indistinguishable from reference); (3) retrain/eval vs 8-substep truth.
Decision: ACCEPT 1 and 2, DROP 3. NEW REQUIREMENT: compute a chaos floor for EACH exp (A, B, C, D, E and each regime/scene type), not just BH and Exp B, so every metric has its own floor to measure against. Not yet applied.

## 3. Optimization
Proposed: (1) adopt kcache,fastio,cl,graph inference stack + cl,graphtrain training stack as defaults; (2) validate or revert `adapt` (ae78f17, job 3326); (3) make `cell` graph-compatible via static candidate cap; (4) park fp16/bf16 training and hand-fused kernels.
Decision: ACCEPT all 4. ADD (5): probe torch.compile/Triton on Polaris (Python.h blocker; options: conda/miniforge env with headers, C_INCLUDE_PATH to matching headers, module avail, uv python). CHECK job 3326 status now (user asked). Not yet applied.
Job 3326 result (checked 2026-10-06): COMPLETED 1m02s. adapt force parity vs eager 9.5e-7 (core0.5+halo6 N=3000), final_pos_sum matches cell. Cold N=1e5: cell 647 ms vs adapt 335 ms (1.9x); warm 1e5: 252 vs 193; but N=1e4 adapt is SLOWER (15.9 vs 3.9 ms warm, 18.5 vs 5.2 cold); N=3e4 warm 20.6 vs 15.3, cold 33.9 vs 29.9. Peak mem lower for adapt (1e5 cold 989 vs 1506 MB). => use adapt only for N >~ 5e4 or dense cores.

## 4. Scaling
Proposed: (1) document zero-shot validity N<=~1e3 cold / ~5e3 warm per 64-unit box; (2) cell-unit distances in kernel/pair term + random-h training (RETRAIN); (3) fine-tune on dense large-N clusters / N-aware density normalisation for N>=1e4; (4) use err@5 not err@20 at N=1e5.
Decision: ACCEPT all 4, item 2 retrain QUEUED (not launched; apply phase). Terminology note: scaling-agent warm/cold = velocity family (vfac 1.0 / 0.3); opt-bench --cold = tighter fixed Gaussian (std 8) density flag.

## 5. Conservation / long-horizon
Proposed: (1) conservative+central force (potential-only / antisymmetric near-field); (2) analytic/learned far-field monopole(+quadrupole) outside grid + edge taper; (3) dE/dL/curl penalties on 100+ tick unrolls; (4) 2 substeps/tick; (5) conservation diagnostics as standing gate rows.
Decision: REJECT 1 as a hard-coded constraint (STANDING PRINCIPLE: do not hard-code conservative/central/antisymmetric forces; the model must be able to learn non-reciprocal/uneven interactions). APPROVE 2, 3, 4, 5. Keep 3 as a soft penalty only. Not yet applied.

## 6. OOD stress
Proposed: (1) fine-tune log-uniform masses 0.1-10 then wider + density normalisation or dt*sqrt(M) input; (2) boundary robustness via far-field monopole from #5 fix 2; (3) eps input + mixed-eps training (0.1-2); (4) fine-tune on dense/small-core clusters (sigma to 1, Plummer a to 0.25; ties to cell-size/random-h retrain from #4); (5) compact OOD battery (~15 points) as non-gating harness report row.
Decision: ACCEPT all 5. STANDING RULE (applies to every fix in this review): each fix is implemented and tested INDIVIDUALLY (one change vs the frozen baseline, own regression-harness + relevant stress rows) so the improvement per fix is attributable. Consequence: do NOT fold fixes 1/3/4 into one combined retrain by default; run separate ablations (combined run only afterwards as a final integration check). Not yet applied.

## 7. Scenes (IN PROGRESS)
Proposed: (1) adopt BounceScatterField + cell-list contact; (2) planetary/orbit after #6 mass fixes; (3) globular after #5/#4/#6; (4) binary-only fine-tune to separate data-mix confound; (5) no full-train-from-scratch for N=300 scenes.
User objection to (1): BounceScatterField needs per-ball distance to each wall; wants something general for ARBITRARY shapes at ARBITRARY positions, and wants ball-ball collisions through the same system; asked if a custom kernel can do it. Direction proposed: obstacles as a rasterized signed-distance/occupancy channel on the existing mesh + learned contact kernel (wall via mesh convolution/SDF gather, ball-ball via pair term with per-body radius); wall force applied after momfix. Decisions on (2)-(5) still pending.
Decision #7 (final): ACCEPT rewritten fix 1 = option B (SDF gather + MLP, solid-filled SDF, stencil, force added after momfix, reserve restitution/material channel slot, moving obstacles out of scope) with option A (kernel convolution of occupancy) as ablation. ACCEPT 2-5 as written (2,3 depend on #6/#5/#4 fixes landing first). Options saved in docs/debugging/obstacle-contact-force-options.md. Not yet applied.

## 8. Scatter vs CF gap
Proposed: (1) analytic-prior residual kernel + analytic mesh-complement PP term (zero-init); (2) pp 2 -> ~8, kNN k 32-48 (needs cell-list/adapt for scale); (3) train >=4000 it; (4) finer mesh G=256/h=.25 (hold); (5) phase-aware mesh correction (park). 1 and 2 are coupled (prior with pp=2 is worse: .0117).
Decision: KEEP the working fix (sf_prior_pp8 = 1+2+3; err@20 .0019 @4000 it vs CF .0025, err@100 .20 vs .29). 4 held, 5 parked (user did not address them; defaulted). Test 1+2 as a pair then ablate each (individual-testing rule); scalability of pp=8 beyond N=100 still untested. Not yet applied.

## 9. Tick-8109 state analysis
Proposed: (1) fix sign in web/scatter/js/truth.js accel, verify truth energy conserved/bound; (2) escaper/boundary policy in web model (= #5 fix 2); (3) log model E and #outside per tick for first 1000 ticks; (4) include seed + meta in state export; (5) investigate faint straight bands in field heatmap (low priority); (6) treat earlier err/truthE readouts from the page as invalid.
Decision: ACCEPT all 6. APPLIED 2026-10-06 except item 2 (deferred to far-field fix #5.2): 1 was already fixed in ff36e31, now covered by tests/truth.mjs; 3 and 4 in the page + tools/log_energy.mjs; 5 reported, 6 banner added. Results in scatter-state-8109-analysis.md "Follow-up 2026-10-06". Earlier err/truthE readouts from the page (before ff36e31) are INVALID.

## 10. Causal field 2-body
Proposed: (1) filter field (smoothed source ramp / filtered stencil) and re-run delete-the-star; (2) isolate early onset via dx/CIC footprint; (3) physically consistent truth (Lienard-Wiechert); (4) no retarded-Newton training target for m>0; (5) benchmark N>2, shared single field; (6) park learned field.
Decision: PINNED. Exploratory/curiosity, not important now. Nothing to apply.

## Walkthrough complete (2026-10-06)
All 10 answered. Apply phase NOT started. Open/unspecified: the user did not say whether #6 fixes 1/3/4 fold into one retrain (resolved by the individual-testing rule: separate runs).
